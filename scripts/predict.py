"""시험 원본 순서의 확률 제출과 이번 달 드론별 최신 기록 상위 20대 선정."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdm_lib import load_records, prepare_records
from train import ART, CAT_COLS, FEATURES, validate_saved_training

REPORTS = ROOT / "reports"
NAMES = {"equipment_type": "드론 유형", "maintenance_type": "정비 구분", "fault_category": "고장 유형",
         "equipment_age_years": "경과연수", "severity": "고장 심각도", "downtime_hours": "운용중단 시간",
         "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수", "maintenance_cost": "정비 비용",
         "maintenance_success": "정비 성공 여부", "cumulative_operating_hours": "누적 비행시간",
         "days_since_last_maintenance": "직전 정비 뒤 경과일", "is_first_maintenance": "최초 기록 표시",
         "fail_count_90d": "최근 90일 고장 수", "days_since_last_failure": "최근 고장 뒤 경과일",
         "hours_since_last_pm": "예방정비 뒤 비행시간"}


def explain(model, frame):
    """LightGBM의 양의 SHAP 기여도를 원래 컬럼으로 합쳐 상위 3개와 값을 출력한다."""
    transformed = model.named_steps["pre"].transform(frame[FEATURES])
    booster = model.named_steps["model"].booster_
    values = booster.predict(transformed, pred_contrib=True)
    np.testing.assert_allclose(values.sum(axis=1), booster.predict(transformed, raw_score=True), atol=1e-7)
    contributions = pd.DataFrame(0., index=range(len(frame)), columns=FEATURES)
    for i, col in enumerate(transformed.columns):
        original = next((cat for cat in CAT_COLS if col.startswith(cat + "_")), col)
        contributions[original] += values[:, i]
    reasons = []
    for i, (_, row) in enumerate(frame.iterrows()):
        positive = contributions.iloc[i].sort_values(ascending=False)
        positive = positive[positive > 0].head(3)
        parts = []
        for col in positive.index:
            value = row[col]
            value = "미상" if pd.isna(value) else f"{value:g}" if isinstance(value, (float, np.floating)) else str(value)
            parts.append(f"{NAMES[col]}({value})")
        reasons.append(", ".join(parts) if parts else "양의 기여 요인 없음")
    return reasons


def calculate():
    card = validate_saved_training()
    prepared = prepare_records()
    model = joblib.load(ART / "drone_model.joblib")
    assert list(model.feature_names_in_) == FEATURES
    test = prepared["test"]
    submission = test[["equipment_id", "maintenance_date"]].copy()
    submission["maintenance_date"] = submission.maintenance_date.dt.strftime("%Y-%m-%d")
    submission["prob"] = model.predict_proba(test[FEATURES])[:, 1]
    raw_test = load_records("test")
    assert len(submission) == len(raw_test) == 1241
    assert submission.equipment_id.tolist() == raw_test.equipment_id.tolist()
    assert submission.maintenance_date.tolist() == raw_test.maintenance_date.dt.strftime("%Y-%m-%d").tolist()
    assert np.isfinite(submission.prob).all() and submission.prob.between(0, 1).all()
    month = prepared["this_month"]
    latest = month.sort_values(["maintenance_date", "equipment_id"], kind="stable").drop_duplicates("equipment_id", keep="last").copy()
    latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    latest = latest.sort_values(["prob", "equipment_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    latest["rank"] = np.arange(1, len(latest) + 1)
    latest["risk_high"] = latest.prob.ge(card["threshold"])
    latest["selected"] = latest["rank"].le(card["capacity"])
    latest["reasons"] = explain(model, latest)
    latest["recommendation"] = "이력·현재 상태 확인 후 재점검"
    assert latest.equipment_id.is_unique and latest.selected.sum() == 20
    assert latest.prob.between(0, 1).all() and latest.prob.is_monotonic_decreasing
    assert (latest.set_index('equipment_id').maintenance_date == month.groupby('equipment_id').maintenance_date.max().reindex(latest.equipment_id).to_numpy()).all()
    keep = ["rank", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "prob", "risk_high", "selected", "reasons", "recommendation"]
    latest = latest[keep]
    latest["maintenance_date"] = latest.maintenance_date.dt.strftime("%Y-%m-%d")
    score = {"period": "2026-01", "records": len(month), "devices": len(latest), "capacity": card["capacity"],
             "threshold": card["threshold"], "high_risk": int(latest.risk_high.sum()),
             "selected": int(latest.selected.sum()),
             "high_risk_unselected": int((latest.risk_high & ~latest.selected).sum()),
             "rows": latest.to_dict("records")}
    return submission, score, card


def fingerprints():
    paths = [ART / "drone_model.joblib", ART / "model_card.json", ROOT / "pdm_lib.py", ROOT / "scripts/predict.py"]
    paths += sorted((ROOT / "data").glob("*.csv"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_outputs():
    manifest = json.loads((ART / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"] == fingerprints(), "예측 근거 변경: predict.py 재실행 필요"
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"예측 파일 변경: {name}"
    expected, score, card = calculate()
    actual = pd.read_csv(REPORTS / "test_predictions_2025.csv")
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_exact=False, rtol=1e-12)
    saved = json.loads((REPORTS / "score_2026-01.json").read_text(encoding="utf-8"))
    assert saved == score, "이번 달 점수 또는 설명 불일치"
    selected = pd.read_csv(REPORTS / "reinspection_2026-01.csv")
    wanted = pd.DataFrame(score["rows"]).loc[lambda d: d.selected].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, wanted, check_exact=False, rtol=1e-12)
    return score


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    before = fingerprints()
    submission, score, card = calculate()
    REPORTS.mkdir(exist_ok=True)
    submission.to_csv(REPORTS / "test_predictions_2025.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "score_2026-01.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = pd.DataFrame(score["rows"]).loc[lambda d: d.selected]
    chosen.to_csv(REPORTS / "reinspection_2026-01.csv", index=False, encoding="utf-8-sig")
    lines = ["# 시험 예측·재점검 대상 선정 (7단계)", "",
             f"- 갱신일: {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d} (Asia/Seoul)",
             "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/predict.py`.",
             "- 근거: artifacts/drone_model.joblib, artifacts/model_card.json, data/drone_*.csv.",
             "- 시험 제출: reports/test_predictions_2025.csv, 원본과 동일 순서의 1,241행, equipment_id·maintenance_date·prob 3열.",
             "- 2025년 정답이 없으므로 시험 성능은 미확인이다. 이번 달은 실습 데이터의 2026년 1월이다.",
             f"- 이번 달 정비 {score['records']}건 / 드론 {score['devices']}대 / 선정 {score['selected']}대 / 위험 높음 {score['high_risk']}대 (기준 {score['threshold']:.0%}).",
             f"- 위험 높음 중 미선정 {score['high_risk_unselected']}대는 담당자가 추가 자원·일정과 현재 상태를 확인한다. 자동으로 다음 달로 미루지 않는다.",
             "- 이전 기간 이력을 이어 시간 피처를 계산하고, 드론별 이번 달 마지막 정비 기록만 남겼다. 확률 동률은 드론 번호 순이다.",
             "", "## 재점검 대상 20대", "",
             "| 순위 | 드론 | 유형 | 부대 | 최근 정비일 | 확률 | 위험을 올린 요인 | 권고 |",
             "|---|---|---|---|---|---:|---|---|"]
    for r in chosen.itertuples(index=False):
        lines.append(f"| {r.rank} | {r.equipment_id} | {r.equipment_type} | {r.unit_code} | {r.maintenance_date} | {r.prob:.1%} | {r.reasons} | {r.recommendation} |")
    lines += ["", "## 해석", "- 선정 유형별: " + ", ".join(f"{k} {v}대" for k,v in Counter(chosen.equipment_type).most_common()) + ".",
              "- 위험 요인은 LightGBM SHAP의 양의 기여도를 원래 컬럼별로 합산한 상위 3개다. 로그 오즈 척도의 예측 기여이며 원인이나 조치 효과를 뜻하지 않는다.",
              "- 검증 AP·월 적중은 이전 기간 평가 모델의 결과다. 이번 파일의 최종 모델이나 2025년 시험 성능으로 오인하지 않는다.",
              "", "## 모델 한계"] + ["- " + text for text in card["limits"]]
    (REPORTS / "selection_2026-01.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert before == fingerprints(), "예측 중 원본 또는 모델 변경"
    outputs = ["test_predictions_2025.csv", "score_2026-01.json", "reinspection_2026-01.csv", "selection_2026-01.md"]
    manifest = {"sources": before, "outputs": {f"reports/{name}": hashlib.sha256((REPORTS/name).read_bytes()).hexdigest() for name in outputs}}
    (ART / "prediction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validate_outputs()
    print(f"시험 예측 {len(submission)}건 / 드론 {score['devices']}대 중 선정 {score['selected']}대 / 위험 높음 {score['high_risk']}대")
    print(chosen[["rank", "equipment_id", "prob"]].head(3).to_string(index=False))
    print("PASS: 시험 원본 순서·전 행 대응·확률 범위·월 최신 기록·20대 중복 없음·재계산 일치")


if __name__ == "__main__":
    main()
