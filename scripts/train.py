"""고정된 드론 모델을 과거→미래로 검증하고 성숙한 정답 기록으로 최종 학습한다."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdm_lib import TARGET, add_time_features, clean, load_records

CAT_COLS = ["equipment_type", "maintenance_type", "fault_category"]
FEATURES = CAT_COLS + ["equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
    "parts_replaced_count", "maintenance_cost", "maintenance_success", "cumulative_operating_hours",
    "days_since_last_maintenance", "is_first_maintenance", "fail_count_90d", "days_since_last_failure", "hours_since_last_pm"]
ART = ROOT / "artifacts"


def make_model():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
                            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    model = LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=4, min_child_samples=60,
                          reg_lambda=5., subsample=.8, subsample_freq=1, colsample_bytree=.8,
                          random_state=42, n_jobs=2, verbose=-1)
    return Pipeline([("pre", pre), ("model", model)])


def pick_threshold(y, prob, miss_cost=300):
    thresholds = np.round(np.arange(.05, .95, .01), 2)
    costs = [int(((prob < t) & (y == 1)).sum()) * miss_cost + int((prob >= t).sum()) * 60 for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


def monthly_backtest(frame):
    rows = []
    for month, group in frame.groupby(frame.maintenance_date.dt.to_period("M")):
        latest = group.sort_values(["maintenance_date", "equipment_id"], kind="stable").drop_duplicates("equipment_id", keep="last")
        k = min(20, len(latest))
        ranked = latest.sort_values(["prob", "equipment_id"], ascending=[False, True], kind="stable").head(k)
        oldest = latest.sort_values(["equipment_age_years", "equipment_id"], ascending=[False, True], kind="stable").head(k)
        rows.append({"month": str(month), "devices": len(latest), "picked": k,
                     "model": int(ranked[TARGET].sum()), "oldest": int(oldest[TARGET].sum()),
                     "random_expected": float(k * latest[TARGET].mean())})
    return pd.DataFrame(rows)


def metrics(frame, threshold):
    y = frame[TARGET].to_numpy().astype(int)
    prob = frame.prob.to_numpy(); predicted = prob >= threshold
    return {"rows": len(frame), "failures": int(y.sum()), "base_rate": float(y.mean()),
            "pr_auc_ap": float(average_precision_score(y, prob)), "roc_auc": float(roc_auc_score(y, prob)),
            "accuracy": float(accuracy_score(y, predicted)), "all_negative_accuracy": float((y == 0).mean()),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "confusion_TN_FP_FN_TP": confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()}


def source_hashes():
    files = [ROOT / "data/drone_train.csv", ROOT / "pdm_lib.py", ROOT / "AGENTS.md",
             ROOT / "scripts/train.py", ROOT / ".agents/skills/pdm-train/SKILL.md"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_saved_training():
    """저장된 평가를 원본 정답·평가 모델로 재계산한다. 파일을 쓰지 않는다."""
    card = json.loads((ART / "model_card.json").read_text(encoding="utf-8"))
    assert card["source_hashes"] == source_hashes(), "학습 근거 변경: 재학습 필요"
    for name, digest in card["artifact_hashes"].items():
        assert hashlib.sha256((ART / name).read_bytes()).hexdigest() == digest, f"산출물 변경: {name}"
    data = add_time_features(clean(load_records("train")))
    valid = data[data.maintenance_date.ge("2024-07-01") & data.maintenance_date.lt("2024-12-01")].copy()
    predictions = pd.read_csv(ART / "validation_predictions.csv", parse_dates=["maintenance_date"])
    pd.testing.assert_frame_equal(valid[["equipment_id", "maintenance_date"]].reset_index(drop=True),
                                  predictions[["equipment_id", "maintenance_date"]])
    np.testing.assert_array_equal(valid[TARGET], predictions[TARGET])
    model = joblib.load(ART / "validation_model.joblib")
    assert list(model.feature_names_in_) == FEATURES == card["features"]
    np.testing.assert_allclose(model.predict_proba(valid[FEATURES])[:, 1], predictions.prob)
    valid["prob"] = predictions.prob.to_numpy()
    threshold = pick_threshold(valid[TARGET].to_numpy(), valid.prob.to_numpy())
    assert threshold == card["threshold"]
    recalculated = metrics(valid, threshold)
    for key, value in recalculated.items():
        np.testing.assert_allclose(value, card["validation"][key])
    back = monthly_backtest(valid)
    saved = pd.read_csv(ART / "backtest_validation.csv")
    pd.testing.assert_frame_equal(back, saved, check_exact=False, rtol=1e-10)
    for col in ["model", "oldest", "random_expected"]:
        np.testing.assert_allclose(back[col].sum(), card["monthly_hits"][col])
    assert (ROOT / "reports/model_validation.md").is_file()
    return card


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = load_records("train")
    data = add_time_features(clean(raw))
    train_mask = data.maintenance_date + pd.Timedelta(days=30) < pd.Timestamp("2024-07-01")
    valid_mask = data.maintenance_date.ge("2024-07-01") & data.maintenance_date.lt("2024-12-01")
    final_mask = data.maintenance_date + pd.Timedelta(days=30) < pd.Timestamp("2025-01-01")
    train, valid = data.loc[train_mask], data.loc[valid_mask].copy()
    model = make_model().fit(train[FEATURES], train[TARGET].astype(int))
    valid["prob"] = model.predict_proba(valid[FEATURES])[:, 1]
    threshold = pick_threshold(valid[TARGET].to_numpy(), valid.prob.to_numpy())
    results = metrics(valid, threshold)
    back = monthly_backtest(valid)
    # 보정 민감도: 다른 전처리·피처·모델은 유지하고 극단값 2개 컬럼만 원값 사용.
    alternative = data.copy()
    for col in ["downtime_hours", "maintenance_cost"]:
        alternative[col] = raw[col]
    alt_model = make_model().fit(alternative.loc[train_mask, FEATURES], train[TARGET].astype(int))
    alt_valid = alternative.loc[valid_mask].copy()
    alt_valid["prob"] = alt_model.predict_proba(alt_valid[FEATURES])[:, 1]
    alt_back = monthly_backtest(alt_valid)
    final = make_model().fit(data.loc[final_mask, FEATURES], data.loc[final_mask, TARGET].astype(int))
    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "validation_model.joblib")
    joblib.dump(final, ART / "drone_model.joblib")
    valid[["equipment_id", "maintenance_date", "equipment_age_years", TARGET, "prob"]].to_csv(
        ART / "validation_predictions.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    back.to_csv(ART / "backtest_validation.csv", index=False, encoding="utf-8-sig")
    card = {"model": "LightGBM / seed 42 / fixed parameters", "features": FEATURES,
            "train_rows": len(train), "train_last_date": str(train.maintenance_date.max().date()),
            "validation_start": "2024-07-01", "validation_end_exclusive": "2024-12-01",
            "final_train_rows": int(final_mask.sum()), "final_last_date": str(data.loc[final_mask].maintenance_date.max().date()),
            "threshold": threshold, "miss_cost": 300, "check_cost": 60, "capacity": 20,
            "validation": results, "monthly_hits": {col: float(back[col].sum()) for col in ["model", "oldest", "random_expected"]},
            "sensitivity": {"raw_extremes_ap": float(average_precision_score(alt_valid[TARGET], alt_valid.prob)),
                            "raw_extremes_hits": int(alt_back.model.sum())},
            "source_hashes": source_hashes(),
            "limits": ["교육용 가상 데이터이며 실제 운용에 그대로 적용하지 않는다.",
                       "검증 기간에서 임계값도 선택했으므로 독립 시험 성능이 아니다. 2025년 시험 정답은 제공되지 않았다.",
                       "극단값 보정 원인은 확정되지 않은 가정이며 같은 드론의 반복 기록이 포함된다.",
                       "월별 최신 정비 기록의 재고장 결과로 평가했으며 실제 점검 효과를 입증하지 않는다.",
                       "확률 보정은 별도로 검증하지 않았으며 최종 판단은 정비 담당자가 한다."]}
    output_files = ["validation_model.joblib", "drone_model.joblib", "validation_predictions.csv", "backtest_validation.csv"]
    card["artifact_hashes"] = {name: hashlib.sha256((ART / name).read_bytes()).hexdigest() for name in output_files}
    (ART / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 드론 모델 학습·검증 보고서", "", "- 갱신일: 2026-11-17 (Asia/Seoul)",
             "- 근거: artifacts/model_card.json, artifacts/validation_predictions.csv, artifacts/backtest_validation.csv.",
             "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/train.py`. 적용 스킬: .agents/skills/pdm-train/SKILL.md.",
             f"- 초기 학습 {len(train)}건, 마지막 정비일 {card['train_last_date']}. 정비 후 30일이 검증 시작 전에 끝나는 기록만 사용했다.",
             f"- 검증: 2024년 7~11월 {len(valid)}건. 최종 모델은 2025년 시작 전에 정답이 확정되는 {card['final_train_rows']}건으로 학습했다.",
             "", "## 검증 성능 (정비 기록 단위)", "", "| 지표 | 결과 | 기준 |", "|---|---:|---|",
             f"| PR-AUC (AP) | {results['pr_auc_ap']:.4f} | 재고장 비율 {results['base_rate']:.4f} |",
             f"| ROC-AUC | {results['roc_auc']:.4f} | 무작위 0.5 |",
             f"| 정확도 | {results['accuracy']:.2%} | 전체 음성 {results['all_negative_accuracy']:.2%} |",
             f"| 정밀도 | {results['precision']:.2%} | 기준 {threshold:.2f} |",
             f"| 재현율 | {results['recall']:.2%} | 같은 검증에서 고른 기준 |",
             "", "## 월 20대 적중 비교 (드론별 월 최신 기록)", "",
             "| 월 | 드론 수 | 선정 | 모델 | 오래된 순 | 무작위 기대 |", "|---|---:|---:|---:|---:|---:|"]
    for r in back.itertuples(index=False):
        lines.append(f"| {r.month} | {r.devices} | {r.picked} | {r.model} | {r.oldest} | {r.random_expected:.2f} |")
    hits = card["monthly_hits"]
    lines += [f"| 합계 | — | {int(back.picked.sum())} | {hits['model']:.0f} | {hits['oldest']:.0f} | {hits['random_expected']:.2f} |",
              "", "## 비용 기준과 처리 가정 민감도",
              f"- 놓친 고장 300만원·점검 60만원에서 검증 비용 최소 기준은 {threshold:.2f}다. 기록별 비용이며 월 20대 제약은 별도다.",
              f"- 보정 적용: AP {results['pr_auc_ap']:.4f}, 월 적중 합계 {hits['model']:.0f}건.",
              f"- 극단값 원값 유지: AP {card['sensitivity']['raw_extremes_ap']:.4f}, 월 적중 합계 {card['sensitivity']['raw_extremes_hits']}건.",
              "- 한 번의 검증 비교이며 보정 원인이 사실임을 증명하지 않는다. 결과에 맞춰 주 모델이나 규칙을 바꾸지 않았다.",
              "", "## 한계"] + ["- " + x for x in card["limits"]]
    (ROOT / "reports/model_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"validation": results, "monthly_hits": hits, "threshold": threshold,
                      "sensitivity": card["sensitivity"], "final_train_rows": card["final_train_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
