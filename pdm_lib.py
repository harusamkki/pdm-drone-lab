"""드론 데이터 정리와 정비 시점까지의 이력 피처. 원본 행과 순서를 보존한다."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = "failure_within_30days"
FAIL_TYPES = ["사후정비", "긴급정비"]
TIME_COLS = ["fail_count_90d", "days_since_last_failure", "hours_since_last_pm"]
COST_CORRECTIONS = {15000: 150, 16200: 162, 16800: 168, 18300: 183, 19000: 190}


def load_records(name="train"):
    """train/test/this_month CSV를 원본 순서로 읽고 정비일을 날짜형으로 변환한다."""
    if name not in {"train", "test", "this_month"}:
        raise ValueError("name은 train, test, this_month 중 하나여야 합니다.")
    return pd.read_csv(DATA / f"drone_{name}.csv", encoding="utf-8-sig", parse_dates=["maintenance_date"])


def clean(df):
    """AGENTS.md의 처리 방침을 복사본에 적용한다. 행·순서·인덱스를 보존한다.

    999는 NaN, 지정된 비용 5종만 100으로 나누고 예방정비의 고장 결측만 N/A로 채운다.
    경과일 결측은 유지하고 is_first_maintenance(0/1)로 표시한다.
    """
    result = df.copy()
    result["downtime_hours"] = result["downtime_hours"].replace(999, np.nan)
    result["maintenance_cost"] = result["maintenance_cost"].replace(COST_CORRECTIONS)
    preventive = result["maintenance_type"].eq("예방정비")
    for col in ["fault_code", "fault_category"]:
        result.loc[preventive & result[col].isna(), col] = "N/A"
        if result.loc[~preventive, col].isna().any():
            warnings.warn(f"{col}: 예방정비 외 결측이 있어 미확인으로 유지합니다.", stacklevel=2)
    if result["maintenance_cost"].gt(10000).any():
        warnings.warn("지정된 보정 목록 외 큰 비용이 있어 확인이 필요합니다.", stacklevel=2)
    result["is_first_maintenance"] = result["days_since_last_maintenance"].isna().astype(int)
    return result


def add_time_features(df):
    """해당 드론의 정비일까지(같은 날짜 포함) 기록만으로 피처 3개를 계산한다.

    fail_count_90d: 정비일-90일 이상, 정비일 이하의 사후·긴급정비 수(이번 정비 포함).
    days_since_last_failure: 가장 최근 고장 정비 이후 일수. 이력이 없으면 NaN.
    hours_since_last_pm: 현재 누적 비행시간 - 가장 최근 예방정비 때 누적 비행시간.
        소수 첫째 자리 반올림. 이번이 예방정비면 0, 예방정비 이력이 없으면 NaN.
    행·순서·인덱스를 보존한다. 동일 날짜의 예방정비가 여러 건이면 입력상 마지막 기록을 기준으로 한다.
    """
    result = df.copy()
    work = df.reset_index(drop=True).copy()
    work["maintenance_date"] = pd.to_datetime(work["maintenance_date"], errors="raise")
    if work[["equipment_id", "maintenance_date"]].isna().any().any():
        raise ValueError("장비 번호와 정비일은 결측일 수 없습니다.")
    counts = np.zeros(len(work), dtype=int)
    days = np.full(len(work), np.nan)
    hours = np.full(len(work), np.nan)
    for _, group in work.groupby("equipment_id", sort=False):
        group = group.sort_values("maintenance_date", kind="stable")
        failures = group[group["maintenance_type"].isin(FAIL_TYPES)]
        preventive = group[group["maintenance_type"].eq("예방정비")]
        dates = group["maintenance_date"].to_numpy()
        fail_dates = failures["maintenance_date"].to_numpy()
        pm_dates = preventive["maintenance_date"].to_numpy()
        end = np.searchsorted(fail_dates, dates, side="right")
        start = np.searchsorted(fail_dates, dates - np.timedelta64(90, "D"), side="left")
        positions = group.index.to_numpy()
        counts[positions] = end - start
        known = end > 0
        days[positions[known]] = (dates[known] - fail_dates[end[known] - 1]) / np.timedelta64(1, "D")
        pm_end = np.searchsorted(pm_dates, dates, side="right")
        known_pm = pm_end > 0
        hours[positions[known_pm]] = np.round(
            group["cumulative_operating_hours"].to_numpy()[known_pm]
            - preventive["cumulative_operating_hours"].to_numpy()[pm_end[known_pm] - 1], 1)
        hours[positions[group["maintenance_type"].eq("예방정비").to_numpy()]] = 0
    result[TIME_COLS[0]] = counts
    result[TIME_COLS[1]] = days
    result[TIME_COLS[2]] = hours
    return result


def prepare_records():
    """3개 파일의 이력을 이어 계산하고 각 파일의 원본 행 순서로 반환한다.

    시험 1,241행을 중복 제거하지 않는다. 이번 달 장비별 최신 기록 선택은 선정 단계에서 한다.
    """
    combined = pd.concat([load_records(name).assign(_source=name)
                          for name in ["train", "test", "this_month"]], ignore_index=True)
    prepared = add_time_features(clean(combined))
    return {name: prepared.loc[prepared["_source"].eq(name)].drop(columns="_source").reset_index(drop=True)
            for name in ["train", "test", "this_month"]}
