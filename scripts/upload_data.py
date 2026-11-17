"""업로드 월별 정비 CSV를 메모리에서 검사·예측한다. 원본·보고서를 저장하지 않는다."""
from io import BytesIO
import warnings

import joblib
import numpy as np
import pandas as pd

from pdm_lib import add_time_features, clean, load_records
from predict import explain
from train import ART, CAT_COLS, FEATURES

REQUIRED = ["equipment_id", "unit_code", "maintenance_date", "fault_code"] + CAT_COLS + [
    "equipment_age_years", "severity", "downtime_hours", "repair_duration_hours", "parts_replaced_count",
    "maintenance_cost", "maintenance_success", "cumulative_operating_hours", "days_since_last_maintenance"]
NUMERIC = [c for c in REQUIRED if c not in ["equipment_id", "unit_code", "maintenance_date", "fault_code"] + CAT_COLS]


def score_upload(content, card):
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("CSV 크기는 20MB 이하여야 합니다.")
    try:
        frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            frame = pd.read_csv(BytesIO(content), encoding="cp949")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV를 UTF-8 또는 CP949 인코딩으로 저장해 주세요.") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError("표 형식의 CSV 파일을 올려주세요.") from exc
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError("필수 컬럼이 없습니다: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV에 정비 기록이 없습니다.")
    frame = frame[REQUIRED].copy()  # 정답·월 가동률 등 추가 컬럼은 사용하지 않는다.
    for col in ["equipment_id", "unit_code"] + CAT_COLS[:2]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col}에 빈 값이 있습니다.")
    dates = pd.to_datetime(frame.maintenance_date, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("maintenance_date는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.")
    frame["maintenance_date"] = dates
    if dates.dt.to_period("M").nunique() != 1:
        raise ValueError("재점검 대상 월 하나의 기록만 올려주세요. 여러 달은 월별로 나눠주세요.")
    if frame.duplicated(["equipment_id", "maintenance_date"]).any():
        raise ValueError("같은 드론·같은 정비일 기록이 중복됩니다. 중복을 확인해 주세요.")
    if not frame.maintenance_type.isin(["예방정비", "사후정비", "긴급정비"]).all():
        raise ValueError("maintenance_type은 예방정비·사후정비·긴급정비 중 하나여야 합니다.")
    for col in NUMERIC:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if (frame[col].notna() & converted.isna()).any() or np.isinf(converted).any():
            raise ValueError(f"{col}에는 유한한 숫자 또는 빈 값만 사용할 수 있습니다.")
        if converted.lt(0).any():
            raise ValueError(f"{col}에 음수가 있습니다. 단위와 입력값을 확인해 주세요.")
        frame[col] = converted
    if frame.cumulative_operating_hours.isna().any():
        raise ValueError("이력 피처 계산에 필요한 cumulative_operating_hours가 비어 있습니다.")
    if not frame.maintenance_success.dropna().isin([0, 1]).all():
        raise ValueError("maintenance_success는 0 또는 1이어야 합니다.")
    if not frame.severity.dropna().between(0, 5).all():
        raise ValueError("severity는 0~5 범위여야 합니다.")
    history = pd.concat([load_records(n) for n in ["train", "test", "this_month"]], ignore_index=True)
    messages = []
    new_devices = set(frame.equipment_id) - set(history.equipment_id)
    if new_devices:
        messages.append(f"기존 이력이 없는 드론 {len(new_devices)}대가 있습니다. 시간 피처는 업로드 이력만으로 계산됩니다.")
    if dates.min() <= pd.Timestamp(card["final_last_date"]):
        messages.append("학습 기간과 겹치는 날짜입니다. 과거 조회용 결과이며 독립적인 모델 평가가 아닙니다.")
    # 겹치는 원본 키는 업로드 기록을 우선해 고장 횟수의 이중 집계를 막는다.
    uploaded_keys = pd.MultiIndex.from_frame(frame[["equipment_id", "maintenance_date"]])
    history_keys = pd.MultiIndex.from_frame(history[["equipment_id", "maintenance_date"]])
    history = history.loc[~history_keys.isin(uploaded_keys)]
    combined = pd.concat([history.assign(_upload=False), frame.assign(_upload=True)], ignore_index=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = add_time_features(clean(combined))
    messages += sorted({str(item.message) for item in caught})
    current = prepared.loc[prepared._upload].sort_values(["maintenance_date", "equipment_id"], kind="stable").drop_duplicates("equipment_id", keep="last").copy()
    model = joblib.load(ART / "drone_model.joblib")
    current["prob"] = model.predict_proba(current[FEATURES])[:, 1]
    current = current.sort_values(["prob", "equipment_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    current["rank"] = np.arange(1, len(current) + 1)
    current["reasons"] = explain(model, current)
    current["maintenance_date"] = current.maintenance_date.dt.strftime("%Y-%m-%d")
    return {"period": str(dates.dt.to_period("M").iloc[0]), "records": len(frame), "devices": len(current),
            "capacity": min(card["capacity"], len(current)), "threshold": card["threshold"],
            "rows": current[["rank", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "prob", "reasons"]].to_dict("records"),
            "warnings": messages, "_raw": frame.copy(), "_prepared": prepared.copy()}
