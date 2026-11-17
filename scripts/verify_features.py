"""4단계: 원본 기록을 직접 필터링해 시간 피처를 검산하고 보고서를 저장한다."""
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdm_lib import TIME_COLS, add_time_features, clean, load_records, prepare_records


def validate():
    """파일을 쓰지 않고 원본 보존·처리·전 행 이력 피처와 경계 사례를 확인한다."""
    raw = {name: load_records(name) for name in ["train", "test", "this_month"]}
    prepared = prepare_records()
    source = pd.concat(raw.values(), ignore_index=True)
    saved = source.copy(deep=True)
    cleaned = clean(source)
    for name, df in raw.items():
        actual = prepared[name]
        assert len(actual) == len(df) and actual.index.equals(df.index)
        pd.testing.assert_frame_equal(actual[["equipment_id", "maintenance_date"]],
                                      df[["equipment_id", "maintenance_date"]])
        pd.testing.assert_series_equal(actual["days_since_last_maintenance"], df["days_since_last_maintenance"])
        assert actual["is_first_maintenance"].tolist() == df["days_since_last_maintenance"].isna().astype(int).tolist()
        np.testing.assert_allclose(actual["downtime_hours"], df["downtime_hours"].mask(df["downtime_hours"].eq(999)), equal_nan=True)
        cost = df["maintenance_cost"].copy().astype(float)
        bad_cost = cost.isin([15000, 16200, 16800, 18300, 19000])
        cost.loc[bad_cost] /= 100
        np.testing.assert_allclose(actual["maintenance_cost"], cost)
        for col in ["fault_code", "fault_category"]:
            expected = df[col].copy()
            expected.loc[df.maintenance_type.eq("예방정비") & expected.isna()] = "N/A"
            pd.testing.assert_series_equal(actual[col], expected)
    all_actual = pd.concat(prepared.values(), ignore_index=True)
    # 본 구현의 이진 탐색과 달리, 각 행의 날짜 범위를 원본에서 직접 걸러 독립 비교한다.
    expected = []
    for row in source.itertuples(index=False):
        past = source[(source.equipment_id == row.equipment_id)
                      & (source.maintenance_date <= row.maintenance_date)]
        failures = past[past.maintenance_type.isin(["사후정비", "긴급정비"])]
        count = int((failures.maintenance_date >= row.maintenance_date - pd.Timedelta(days=90)).sum())
        last_failure = failures.maintenance_date.max()
        days = (row.maintenance_date - last_failure).days if pd.notna(last_failure) else np.nan
        pm = past[past.maintenance_type.eq("예방정비")].sort_values("maintenance_date", kind="stable")
        hours = 0.0 if row.maintenance_type == "예방정비" else (
            round(row.cumulative_operating_hours - pm.cumulative_operating_hours.iloc[-1], 1) if len(pm) else np.nan)
        expected.append([count, days, hours])
    np.testing.assert_allclose(all_actual[TIME_COLS], np.array(expected), equal_nan=True, atol=1e-8)
    pd.testing.assert_frame_equal(source, saved)
    assert int(cleaned.downtime_hours.isna().sum()) == 4
    assert int(cleaned.is_first_maintenance.sum()) == 150
    # 정답값 변경은 피처에 영향을 주지 않아야 한다.
    flipped = source.copy()
    flipped["failure_within_30days"] = 1
    pd.testing.assert_frame_equal(add_time_features(clean(flipped))[TIME_COLS], all_actual[TIME_COLS])
    edge = pd.DataFrame({"equipment_id": ["A", "A", "B", "A", "A"],
                         "maintenance_date": pd.to_datetime(["2024-04-01", "2024-01-02", "2024-04-01", "2024-04-01", "2024-04-02"]),
                         "maintenance_type": ["사후정비", "긴급정비", "예방정비", "예방정비", "긴급정비"],
                         "cumulative_operating_hours": [20., 10., 100., 20., 22.]}, index=[9, 2, 9, 1, 8])
    edge_saved = edge.copy(deep=True)
    actual = add_time_features(edge)
    assert actual.index.equals(edge.index)
    assert actual.fail_count_90d.tolist() == [2, 1, 0, 2, 2]
    assert actual.days_since_last_failure.iloc[0] == 0 and pd.isna(actual.days_since_last_failure.iloc[2])
    assert actual.hours_since_last_pm.iloc[0] == 0 and actual.hours_since_last_pm.iloc[-1] == 2
    assert pd.isna(actual.hours_since_last_pm.iloc[1])
    pd.testing.assert_frame_equal(edge, edge_saved)
    # 미래 날짜를 제외해도 그 이전 피처가 그대로인지 확인한다.
    cutoff = pd.Timestamp("2025-01-01")
    before = source[source.maintenance_date < cutoff]
    calculated = add_time_features(clean(before))
    pd.testing.assert_frame_equal(calculated[TIME_COLS], all_actual.loc[before.index, TIME_COLS])
    sample = prepared["train"].loc[lambda d: d.equipment_id.eq("DR-0051") & d.fail_count_90d.gt(0)].sort_values("maintenance_date").iloc[-1]
    start = sample.maintenance_date - pd.Timedelta(days=90)
    records = source[source.equipment_id.eq(sample.equipment_id)
                     & source.maintenance_date.between(start, sample.maintenance_date)].sort_values("maintenance_date")
    return raw, prepared, sample, records


def fingerprint_lines():
    return [f"- {p.relative_to(ROOT).as_posix()}: {hashlib.sha256(p.read_bytes()).hexdigest()}"
            for p in [ROOT / "pdm_lib.py", ROOT / "AGENTS.md", ROOT / "scripts/verify_features.py"]
            + list(sorted((ROOT / "data").glob("*.csv")))]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = list((ROOT / "data").glob("*.csv"))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    raw, prepared, sample, records = validate()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    lines = ["# 4단계 데이터 처리·피처 생성 검증", "",
             f"- 확인일: {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d} (Asia/Seoul)",
             "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/verify_features.py`",
             "- 근거: data/drone_*.csv, pdm_lib.py, AGENTS.md. 원본 SHA256 유지 확인 완료.",
             "- 검증 통과: 원본 보존, 비용·결측 보정, 전체 행 직접 검산, 90일 경계, 같은 날짜, 정비 파일 간 이력 연결, 장비 분리, 중복 인덱스, 미래 기록·정답 변경의 영향 없음.",
             "", "## 데이터별 결과", "", "| 파일 | 행 수 | 고장 횟수 합 | 직전 고장 일수 결측 | 예방정비 후 시간 결측 |", "|---|---:|---:|---:|---:|"]
    for name, df in prepared.items():
        lines.append(f"| {name} | {len(df)} | {int(df.fail_count_90d.sum())} | {int(df.days_since_last_failure.isna().sum())} | {int(df.hours_since_last_pm.isna().sum())} |")
    start = sample.maintenance_date - pd.Timedelta(days=90)
    count = int(records.maintenance_type.isin(["사후정비", "긴급정비"]).sum())
    assert count == sample.fail_count_90d
    lines += ["", "## 표본 직접 검산", "",
              f"- 장비: {sample.equipment_id}, 정비일: {sample.maintenance_date:%Y-%m-%d}.",
              f"- 최근 90일 범위: {start:%Y-%m-%d} ~ {sample.maintenance_date:%Y-%m-%d} (양 끝 포함).",
              f"- 원본 기록에서 직접 센 고장 정비: {count}건 / 구현한 값: {int(sample.fail_count_90d)}건. 일치한다.",
              "", "| 정비일 | 정비 유형 | 고장 횟수 포함 |", "|---|---|---|"]
    for r in records.itertuples():
        lines.append(f"| {r.maintenance_date:%Y-%m-%d} | {r.maintenance_type} | {'포함' if r.maintenance_type in ['사후정비', '긴급정비'] else '제외'} |")
    lines += ["", "## 검증 시점의 파일 지문", ""] + fingerprint_lines()
    path = ROOT / "reports/feature_verification.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"검증 통과. 저장: {path}")
    print(f"표본 {sample.equipment_id} {sample.maintenance_date:%Y-%m-%d}: 직접 집계 {count} / 구현 {int(sample.fail_count_90d)}")


if __name__ == "__main__":
    main()
