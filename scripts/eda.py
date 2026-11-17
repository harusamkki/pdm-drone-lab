"""원본 CSV 3개를 읽어 탐색 결과를 reports/eda.md에 기록한다. 원본은 변경하지 않는다."""
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET = "failure_within_30days"


def table(frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(cell(v) for v in row) + " |"
             for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def rates(frame, key):
    result = frame.groupby(key, dropna=False)[TARGET].agg(건수="size", 재고장="sum")
    result["재고장 비율(%)"] = result["재고장"] / result["건수"] * 100
    return result.reset_index()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = [ROOT / "data" / f"drone_{name}.csv" for name in ["train", "test", "this_month"]]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    datasets = {p.stem: pd.read_csv(p, encoding="utf-8-sig") for p in paths}
    lines = ["# 드론 예측정비 — 데이터 탐색 보고서", "",
             f"- 갱신일: {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d} (Asia/Seoul)",
             "- 변경 요약: 원본 데이터의 구조·결측·분포·재고장 패턴을 최초 집계했다.",
             "- 근거: data/drone_train.csv, data/drone_test.csv, data/drone_this_month.csv.",
             "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/eda.py` (프로젝트 폴더).",
             "- 원본 자동 추론 자료형 기준이다. 날짜는 집계용 복사본에서만 변환한다.",
             "- 모든 비율은 장비 수가 아닌 정비 기록 건수 기준이다. 교육용 가상 데이터이며 인과관계를 의미하지 않는다.",
             "", "## 1. 파일 구성"]
    summaries = []
    for name, df in datasets.items():
        dates = pd.to_datetime(df["maintenance_date"], errors="coerce")
        summaries.append([name, len(df), len(df.columns), df["equipment_id"].nunique(),
                          dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d"),
                          "있음" if TARGET in df else "없음", int(df.duplicated().sum()),
                          int(df.duplicated(["equipment_id", "maintenance_date"]).sum()), int(dates.isna().sum())])
    lines += [table(pd.DataFrame(summaries, columns=["파일", "행 수", "컬럼 수", "드론 수", "시작", "종료", "정답", "완전 중복", "장비·날짜 중복", "날짜 오류"]))]
    for name, df in datasets.items():
        lines += ["", f"## 2. {name} 자료형·결측", table(pd.DataFrame({
            "컬럼": df.columns, "자료형": df.dtypes.astype(str).values,
            "결측 수": df.isna().sum().values, "결측 비율(%)": df.isna().mean().values * 100}))]
        lines += ["", "### 수치 분포", table(df.select_dtypes("number").describe(
            percentiles=[.25, .5, .75]).T.reset_index().rename(columns={"index": "컬럼"}))]
        outliers = []
        for col in df.select_dtypes("number"):
            if col == TARGET or df[col].nunique() <= 2:
                continue
            s = df[col]; q1, q3 = s.quantile([.25, .75]); spread = q3 - q1
            lo, hi = q1 - 1.5 * spread, q3 + 1.5 * spread
            outliers.append([col, lo, hi, int(((s < lo) | (s > hi)).sum()), int((s < 0).sum())])
        lines += ["", "### IQR 이상치 후보", "Q1−1.5×IQR 미만 또는 Q3+1.5×IQR 초과인 기록이다. 오류라고 확정하거나 자동 보정하지 않는다.",
                  table(pd.DataFrame(outliers, columns=["컬럼", "하한", "상한", "후보 건수", "음수 건수"]))]
        for col in ["downtime_hours", "maintenance_cost"]:
            lines += ["", f"### {col} 상위 10건 (전체 후보 목록이 아님)",
                      table(df.nlargest(10, col)[["equipment_id", "maintenance_date", "maintenance_type", col]])]
        lines += ["", "### 결측의 문맥"]
        for col in ["fault_code", "fault_category"]:
            subset = df[df[col].isna()]
            counts = subset.groupby("maintenance_type", dropna=False).size().reset_index(name="결측 건수")
            lines += [f"- {col} 결측의 정비 유형별 분포", table(counts)]
        missing = df[df["days_since_last_maintenance"].isna()]
        dates = pd.to_datetime(df["maintenance_date"])
        first = dates.groupby(df["equipment_id"]).transform("min")
        lines += [f"- 정비 경과일 결측 {len(missing)}건, 해당 드론 {missing['equipment_id'].nunique()}대. "
                  f"파일 내 해당 드론의 가장 이른 날짜와 일치한 결측은 {int((dates.eq(first) & df['days_since_last_maintenance'].isna()).sum())}건이다. "
                  "파일 내 최초 기록이라는 사실만으로 실제 생애 첫 정비라고 단정하지 않는다."]
    train = datasets["drone_train"].copy()
    assert train[TARGET].notna().all() and train[TARGET].isin([0, 1]).all()
    lines += ["", "## 3. 학습 데이터의 재고장 비율",
              f"- 전체 {len(train):,}건 중 재고장 {int(train[TARGET].sum()):,}건, 비율 {train[TARGET].mean():.2%}.",
              f"- 전부 재고장 없음으로 예측한 정확도는 {1-train[TARGET].mean():.2%}다. 모델 성능이 아니다."]
    for key in ["maintenance_type", "equipment_type", "fault_category"]:
        lines += ["", f"### {key}별", table(rates(train, key))]
    dates = pd.to_datetime(train["maintenance_date"])
    train["월"] = dates.dt.month
    train["연월"] = dates.dt.strftime("%Y-%m")
    train["계절"] = dates.dt.month.map({12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"})
    for key in ["계절", "월", "연월"]:
        lines += ["", f"### {key}별", table(rates(train, key))]
    lines += ["", "### 드론 유형 × 고장 유형의 정비 건수 (결측 별도 표시)",
              table(pd.crosstab(train["equipment_type"], train["fault_category"].fillna("결측")).reset_index()),
              "", "## 4. 다음 단계에서 판단할 사항",
              "- fault_code·fault_category의 결측이 예방정비의 고장 없음인지 위 정비 유형별 표를 보고 결정한다.",
              "- 운용중단 시간·비용의 극단값은 원본 값을 보존한 채 입력 코드 또는 단위 오류 여부를 확인한다. 이전 실습의 보정 규칙을 자동 적용하지 않는다.",
              "- 경과일 결측은 이전 파일의 정비 이력과 이어서 확인해야 한다. 파일 경계에서 실제 첫 정비로 오인하지 않는다.",
              "- availability_rate는 그 달 전체 가동률이므로 정비 직후에 알 수 없는 누수 후보이며 이번 달 파일에도 없다. 예측 입력에서 제외할 근거로 기록한다.",
              "- 드론 유형·정비 유형·계절의 비율 차이는 구성비와 표본 수의 영향을 받는다. 원인이나 미래 성능으로 단정하지 않는다.",
              "- 시험·이번 달에는 정답이 없으므로 재고장 비율이나 모델 성능을 계산하지 않았다.",
              "- 위 표는 원본 집계다. 데이터 처리·피처 검증의 진행 상태는 아래 검증 기록을 따르며 모델 학습·예측·대상 선정은 별도 단계다."]
    rules_path = ROOT / "AGENTS.md"
    if rules_path.exists():
        rules = rules_path.read_text(encoding="utf-8")
        section = re.search(r"(?ms)^### 데이터 처리\s*\n(.*?)(?=^### |\Z)", rules)
        if section and any(line.startswith("- ") for line in section.group(1).splitlines()):
            lines += ["", "## 5. 확정한 처리 방침 (3단계)",
                      "근거: AGENTS.md의 데이터 처리 규칙. 위 표는 보정 전 원본 집계이며 처리 구현과 검증은 다음 절에서 구분한다.",
                      section.group(1).strip()]
            lines[3] = "- 변경 요약: 원본 탐색 결과를 갱신하고 3단계의 결측·극단값 처리 가정과 이력 계산 규칙을 반영했다."
    feature_report = ROOT / "reports/feature_verification.md"
    if feature_report.exists():
        lines += ["", "## 6. 데이터 처리·피처 생성 검증 (4단계)",
                  "다음은 reports/feature_verification.md의 실행 기록이다. 현재 코드·원본과의 일치 여부는 scripts/check.py에서 재검사한다.",
                  feature_report.read_text(encoding="utf-8").replace("# 4단계 데이터 처리·피처 생성 검증", "### 검증 결과", 1)]
        lines[3] = "- 변경 요약: 원본 EDA와 처리 방침을 보존하고 4단계의 피처 구현·전 행 검산·표본 검산 기록을 추가했다."
    if feature_report.exists():
        from pdm_lib import prepare_records
        prepared = prepare_records()
        lines += ["", "## 7. EDA 해석·처리 방침·구현 검토", "",
                  "### EDA 수치의 의미",
                  f"- 학습 기록 {len(train):,}건 중 재고장은 {int(train[TARGET].sum()):,}건({train[TARGET].mean():.2%})이다. 동일 드론의 반복 정비가 포함되므로 드론별 고장 확률이나 독립 표본 수로 해석하지 않는다.",
                  "- 정비 유형별 비율은 정비 직후 재고장과의 연관성을 나타낸다. 긴급정비 자체가 재고장을 일으킨다는 인과 해석은 할 수 없다.",
                  "- 유형·계절별 비율 차이는 아래 건수와 함께 본다. 정비 유형, 운용량, 드론별 반복 기록 등 구성 차이를 통제하지 않은 탐색 결과이며 유의성이나 일반화 성능을 검증한 결과가 아니다.",
                  table(rates(train, "equipment_type")), table(rates(train, "계절")),
                  "- fault_code·fault_category 결측은 세 파일 모두 예방정비에서만 발생했다. 정상 정비에서 고장 정보가 없는 문맥으로 해석하는 방침은 현재 데이터와 일치한다.",
                  "- 경과일 결측 150건은 제공 이력의 첫 기록이다. 이력 시작 전에 정비가 없었다는 뜻은 아니므로, 표시 이름과 달리 생애 첫 정비로 해석하지 않는다.",
                  "### 처리 방침과 실제 결과",
                  "- 999→결측과 지정 비용 5종÷100은 입력 오류를 확인한 사실이 아니라 분석용 가정이다. 담당자 확인 전에는 확정 원인으로 표현하지 않는다. 이후 검증 기간에서 원값 유지 등 대안과 성능 민감도를 비교할 수 있다.",
                  "- 보정 후 학습 비용 최댓값이 478인 것은 처리 결과 확인이며 모델이 더 좋아졌다는 증거는 아니다.",
                  "- 시험·이번 달의 수치를 품질 점검에만 사용했고 보정 규칙은 학습 데이터의 극단값에서 정했다. 아직 모델 선택이나 시험 정답을 사용하지 않았다.",
                  "| 파일 | 보정 후 비용 최대 | 운용중단 결측 | 최초 기록 표시 합 | 예방정비 후 시간 음수 |",
                  "|---|---:|---:|---:|---:|"]
        for name, frame in prepared.items():
            lines.append(f"| {name} | {frame.maintenance_cost.max()} | {int(frame.downtime_hours.isna().sum())} | {int(frame.is_first_maintenance.sum())} | {int(frame.hours_since_last_pm.lt(0).sum())} |")
        lines += ["", "### 구현 검증 범위와 남은 확인",
                  "- scripts/check.py는 파일 지문을 대조한 뒤 시간 피처를 다시 계산하고 전체 행의 원본 날짜 필터 집계와 비교한다. 비용·결측 처리, 행 순서, 날짜 경계, 미래·정답 정보 배제도 검사한다.",
                  "- DR-0051의 2023-09-11 정비: 2023-06-13~09-11에 07-10 사후정비 1건만 포함된다. 최근 고장 이후 63일이며 당일이 예방정비여서 예방정비 후 비행시간은 0이다.",
                  "- 같은 날짜 기록을 함께 쓰는 것은 현재 실습 규칙이다. 실제 업무에서 하루 중 순서가 중요하면 정비 완료 시각으로 구분해야 한다.",
                  "- 정답이 있는 기간 안에서 학습·검증을 나누고, 30일 정답이 경계 시점에 확정됐는지 확인하는 절차는 5단계에서 정한다.",
                  "- 현재 확인 범위에서 수치·처리 구현의 불일치는 발견하지 못했다. 업무 가정의 사실 여부나 모델의 유용성이 검증됐다는 의미는 아니다."]
        lines[3] = "- 변경 요약: EDA 수치의 의미, 처리 가정과 한계, 보정 후 실제 집계 및 피처 검증 범위를 검토해 추가했다."
    model_report = ROOT / "reports/model_validation.md"
    if model_report.exists():
        lines += ["", "## 8. 모델 학습·검증 결과 (5단계)",
                  "근거: scripts/train.py 실행과 artifacts/model_card.json. 아래는 정답 있는 학습 CSV 내부 검증 결과이며 2025년 시험 성능이 아니다.",
                  model_report.read_text(encoding="utf-8").replace("# 드론 모델 학습·검증 보고서", "### 학습 실행 기록", 1)]
        lines[3] = "- 변경 요약: 기존 탐색·처리·피처 검증을 보존하고 5단계 학습 스킬, 날짜 분리, 검증 성능·기준선과 보정 민감도 결과를 반영했다."
    threshold_report = ROOT / "reports/threshold_whatif.md"
    if threshold_report.exists():
        lines += ["", "## 9. 비용별 임계값 비교 (6단계)",
                  "근거: scripts/threshold_whatif.py. 현재 모델의 임계값은 그대로 유지한 비교 실험이다.",
                  threshold_report.read_text(encoding="utf-8").replace("# 비용 가정에 따른 임계값 비교 (6단계)", "### 비용 비교 실행 기록", 1)]
        lines[3] = "- 변경 요약: 기존 분석을 보존하고 검증 예측의 비용별 임계값 비교와 운영 해석을 추가했다."
    selection_report = ROOT / "reports/selection_2026-01.md"
    if selection_report.exists():
        lines += ["", "## 10. 시험 예측·재점검 선정 (7단계)",
                  "근거: scripts/predict.py, reports/test_predictions_2025.csv, reports/score_2026-01.json.",
                  selection_report.read_text(encoding="utf-8").replace("# 시험 예측·재점검 대상 선정 (7단계)", "### 예측 실행 기록", 1)]
        lines[3] = "- 변경 요약: 기존 분석을 보존하고 2025년 시험 예측과 2026년 1월 재점검 20대 선정 결과를 추가했다."
    if (ROOT / "reports/analysis_report.md").exists():
        lines += ["", "## 11. 제출용 보고서·검수 (8단계)",
                  "- 분석 보고서: reports/analysis_report.md.",
                  "- 재점검 계획: reports/2026-01_재점검계획.md.",
                  "- 검수 기록: reports/verification.md. 현재 결과와의 재대조는 scripts/check.py로 수행한다.",
                  "- 시험 제출 파일: reports/test_predictions_2025.csv. 생성만 완료됐으며 강사에게 실제 제출한 것은 아니다."]
        lines[3] = "- 변경 요약: 1~7단계의 상세 근거를 보존하고 8단계 제출 보고서·재점검 계획·숫자 검수 결과의 위치를 추가했다."
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    lines += ["", "## 원본 보존 확인", "실행 전후 SHA256 일치 확인 완료.",
              table(pd.DataFrame([[p.name, h] for p, h in hashes.items()], columns=["파일", "SHA256"]))]
    out = ROOT / "reports" / "eda.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out}")
    print(f"학습: {len(train)}건, 재고장 {int(train[TARGET].sum())}건, 비율 {train[TARGET].mean():.2%}")
    for name, df in datasets.items():
        print(f"{name}: 결측 {df.isna().sum()[df.isna().sum() > 0].to_dict()}")
    print(table(rates(train, "maintenance_type")))
    print(table(rates(train, "equipment_type")))
    print("원본 3개 SHA256 보존 확인 완료")


if __name__ == "__main__":
    main()
