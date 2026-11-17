"""대시보드 추가 분석. 학습·저장 결과를 변경하지 않는 메모리 집계."""
import joblib
import numpy as np
import pandas as pd

from pdm_lib import load_records, prepare_records, clean
from train import ART, FEATURES, CAT_COLS


def context(score):
    if "_prepared" in score:
        raw, history = score["_raw"].copy(), score["_prepared"].copy()
    else:
        raw = load_records("this_month")
        history = pd.concat(prepare_records().values(), ignore_index=True)
    history = history[history.maintenance_date <= pd.Period(score["period"]).end_time].copy()
    return raw, history


def concentration(ranked, group):
    result = ranked.groupby(group, dropna=False).agg(
        정비드론수=("equipment_id", "size"), 위험높음=("위험 높음", "sum"), 재점검대상=("재점검", "sum"))
    result["위험 비율(%)"] = result["위험높음"] / result["정비드론수"] * 100
    return result.reset_index().sort_values(["위험 비율(%)", "정비드론수"], ascending=False).reset_index(drop=True)


def capacity_comparison(ranked, threshold, current_capacity):
    ordered = ranked.sort_values(["prob", "equipment_id"], ascending=[False, True])
    high = ordered.prob.ge(threshold)
    values = sorted({min(k, len(ordered)) for k in [10, 20, 30, current_capacity, len(ordered)]})
    total = int(high.sum())
    return pd.DataFrame([{"점검 가능 대수": k, "선정 수": k, "위험 드론 포함": int(high.iloc[:k].sum()),
                          "미선정 위험 드론": int(high.iloc[k:].sum()),
                          "기준 미만 포함": int((~high.iloc[:k]).sum()),
                          "위험 드론 포함률(%)": (float(high.iloc[:k].sum()) / total * 100) if total else np.nan}
                         for k in values])


def monthly_history(history, period):
    # 화면 비교에만 최종 모델을 동일 적용한다. 과거 월 독립 성능 평가가 아니다.
    end = pd.Period(period, freq="M")
    start = end - 5
    frame = history[history.maintenance_date.dt.to_period("M").between(start, end)].copy()
    if "_upload" in frame:
        frame = frame.loc[frame.maintenance_date.dt.to_period("M").ne(end) | frame["_upload"]].copy()
    frame["month"] = frame.maintenance_date.dt.strftime("%Y-%m")
    latest = frame.sort_values(["maintenance_date", "equipment_id"], kind="stable").drop_duplicates(["equipment_id", "month"], keep="last").copy()
    if not latest.empty:
        model = joblib.load(ART / "drone_model.joblib")
        latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    else:
        latest["prob"] = pd.Series(dtype=float)
    return latest


def month_changes(ranked, monthly, period):
    previous = str(pd.Period(period, freq="M") - 1)
    prior = monthly.loc[monthly.month.eq(previous), ["equipment_id", "prob", "fail_count_90d"]].rename(
        columns={"prob": "previous_prob", "fail_count_90d": "previous_fail_count_90d"})
    current = ranked[["equipment_id", "prob"]].merge(prior, on="equipment_id", how="left", validate="one_to_one")
    current_counts = monthly.loc[monthly.month.eq(period), ["equipment_id", "fail_count_90d"]]
    current = current.merge(current_counts, on="equipment_id", how="left", validate="one_to_one")
    current["증감(%p)"] = (current.prob - current.previous_prob) * 100
    current["최근 90일 고장 수 증감"] = current.fail_count_90d - current.previous_fail_count_90d
    return current.sort_values("증감(%p)", ascending=False, na_position="last")


def quality(raw, history):
    baseline = clean(load_records("train"))
    missing = pd.DataFrame({"컬럼": raw.columns, "결측 수": raw.isna().sum().values,
                            "결측 비율(%)": raw.isna().mean().values * 100})
    ranges = []
    for col in baseline.select_dtypes("number"):
        if col == "failure_within_30days" or col not in raw:
            continue
        s = pd.to_numeric(raw[col], errors="coerce")
        low, high = baseline[col].min(), baseline[col].max()
        ranges.append({"컬럼": col, "학습 최솟값": low, "학습 최댓값": high,
                       "학습 범위 밖 건수": int((s.lt(low) | s.gt(high)).sum())})
    unknown = []
    for col in CAT_COLS:
        values = sorted(set(raw[col].dropna().astype(str)) - set(baseline[col].dropna().astype(str)))
        unknown.append({"컬럼": col, "새 범주": ", ".join(values) if values else "없음"})
    first_date = raw.groupby("equipment_id").maintenance_date.min()
    prior = {drone: bool(((history.equipment_id == drone) & (history.maintenance_date < date)).any())
             for drone, date in first_date.items()}
    no_history = [drone for drone, exists in prior.items() if not exists]
    return {"missing": missing, "ranges": pd.DataFrame(ranges), "unknown": pd.DataFrame(unknown),
            "no_history": no_history, "duplicates": int(raw.duplicated(["equipment_id", "maintenance_date"]).sum()),
            "sentinel": int(raw.downtime_hours.eq(999).sum()),
            "cost_candidates": int(raw.maintenance_cost.isin([15000, 16200, 16800, 18300, 19000]).sum())}
