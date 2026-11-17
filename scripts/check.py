"""드론 실습 1~8단계 진행 확인. 파일을 변경하거나 학습을 실행하지 않는다."""
import csv
import hashlib
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("드론 예측정비 실습 — 단계별 진행 확인")
    print(f"Python: {sys.executable}")
    problems = []
    for module in ["pandas", "numpy", "sklearn", "lightgbm", "matplotlib", "streamlit", "joblib"]:
        try:
            importlib.import_module(module)
        except Exception as exc:
            problems.append(f"{module}: {type(exc).__name__}: {exc}")
    print("[확인] 필수 라이브러리 있음" if not problems else "[미완료] 라이브러리 설치 필요")
    for problem in problems:
        print(f"  - {problem}")
    ok = not problems
    for name, expected, has_target in [
        ("drone_train.csv", 2346, True),
        ("drone_test.csv", 1241, False),
        ("drone_this_month.csv", 106, False),
    ]:
        path = ROOT / "data" / name
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                count = sum(1 for _ in reader)
            valid = count == expected and ("failure_within_30days" in columns) == has_target
            ok &= valid
            print(f"[{'확인' if valid else '미완료'}] {name}: {count:,}건, "
                  f"정답 {'있음' if 'failure_within_30days' in columns else '없음'}")
        except (OSError, UnicodeError, csv.Error) as exc:
            ok = False
            print(f"[미완료] {name}: {exc}")
    rules_exist = (ROOT / "AGENTS.md").is_file()
    ok &= rules_exist
    print("[확인] AGENTS.md 있음" if rules_exist else "[미완료] AGENTS.md 없음")
    rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8") if rules_exist else ""
    basic = re.search(r"(?ms)^### 기본\s*\n(.*?)(?=^### |\Z)", rules)
    rule_count = sum(line.strip().startswith("- ") for line in basic.group(1).splitlines()) if basic else 0
    stage1 = ok and rule_count >= 3
    eda_path = ROOT / "reports" / "eda.md"
    eda = eda_path.read_text(encoding="utf-8") if eda_path.exists() else ""
    fresh = bool(eda) and all(
        path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() in eda
        for path in [ROOT / "data" / name for name in
                     ["drone_train.csv", "drone_test.csv", "drone_this_month.csv"]]
    )
    stage2 = ok and (ROOT / "scripts" / "eda.py").is_file() and fresh
    stages = [
        ("환경·기본 규칙 준비", "완료" if stage1 else "진행",
         f"라이브러리・원본 데이터 확인 / 기본 규칙 {rule_count}줄"),
        ("데이터 탐색 (EDA)", "완료" if stage2 else "진행" if eda else "대기",
         "탐색 코드・보고서 있음 / 원본 3개 SHA256 일치" if stage2 else
         "scripts/eda.py와 reports/eda.md 확인 필요 (원본 변경 시 EDA 재실행)"),
    ]
    future = [
        ("데이터 처리 방침", "처리 규칙을 AGENTS.md에 작성", []),
        ("피처 생성・표본 검산", "공통 함수 구현 및 드론 1대 수동 검산", ["pdm_lib.py"]),
        ("학습 스킬・모델 검증", "학습 절차, 기간 분리, 검증 지표와 기준선 확인", ["scripts/train.py"]),
        ("비용별 임계값 비교", "놓친 고장 비용 300・600만원 비교", ["reports/threshold_whatif.md"]),
        ("시험 예측・재점검 선정", "2025년 1,241건 예측 및 2026년 1월 드론 20대 선정",
         ["reports/test_predictions_2025.csv"]),
        ("보고서 작성・숫자 검수", "보고서 수치와 실제 출력 대조", ["reports/verification.md"]),
    ]
    processing = re.search(r"(?ms)^### 데이터 처리\s*\n(.*?)(?=^### |\Z)", rules)
    has_processing = bool(processing) and any(
        line.strip().startswith("- ") for line in processing.group(1).splitlines())
    processing_text = processing.group(1) if processing else ""
    processing_count = sum(line.strip().startswith("- ") for line in processing_text.splitlines())
    policy_complete = stage2 and processing_count >= 6 and all(
        field in processing_text for field in ["downtime_hours", "maintenance_cost", "fault_code",
                                               "fault_category", "days_since_last_maintenance",
                                               "is_first_maintenance", "availability_rate"]
    ) and "## 5. 확정한 처리 방침 (3단계)" in eda
    feature_verified = False
    feature_detail = ""
    feature_report = ROOT / "reports/feature_verification.md"
    if feature_report.exists() and (ROOT / "pdm_lib.py").exists():
        try:
            from verify_features import fingerprint_lines, validate
            recorded = feature_report.read_text(encoding="utf-8")
            if not all(line in recorded for line in fingerprint_lines()):
                raise ValueError("코드·규칙·원본 변경: verify_features.py 재실행 필요")
            validate()
            feature_verified = policy_complete
            feature_detail = "처리 검증·전체 행 직접 검산·표본 검산 통과 / 원본·코드 지문 일치"
        except Exception as exc:
            feature_detail = f"검증 실패: {type(exc).__name__}: {exc}"
    training_verified = False
    training_detail = ""
    if (ROOT / "artifacts/model_card.json").exists() and (ROOT / "scripts/train.py").exists():
        try:
            from train import validate_saved_training
            card = validate_saved_training()
            training_verified = feature_verified
            training_detail = (f"검증 AP {card['validation']['pr_auc_ap']:.4f} / 월 적중 "
                               f"{card['monthly_hits']['model']:.0f} 대 오래된 순 {card['monthly_hits']['oldest']:.0f} / "
                               f"기준 {card['threshold']:.2f} / 원본·모델 재계산 일치")
        except Exception as exc:
            training_detail = f"학습 결과 검증 실패: {type(exc).__name__}: {exc}"
    threshold_verified = False
    threshold_detail = ""
    if (ROOT / "reports/threshold_whatif.md").exists():
        try:
            from threshold_whatif import validate_report
            comparisons = validate_report()
            threshold_verified = training_verified
            threshold_detail = " / ".join(f"{cost}만원 → {threshold:.2f}" for cost, threshold, *_ in comparisons)
        except Exception as exc:
            threshold_detail = f"비용 비교 검증 실패: {type(exc).__name__}: {exc}"
    prediction_verified = False
    prediction_detail = ""
    if (ROOT / "artifacts/prediction_manifest.json").exists():
        try:
            from predict import validate_outputs
            score = validate_outputs()
            prediction_verified = threshold_verified
            prediction_detail = (f"시험 1,241행 재계산 일치 / 드론 {score['devices']}대 중 {score['selected']}대 선정 / "
                                 f"위험 높음 {score['high_risk']}대")
        except Exception as exc:
            prediction_detail = f"예측 검증 실패: {type(exc).__name__}: {exc}"
    report_verified = False
    report_detail = ""
    if (ROOT / "artifacts/report_manifest.json").exists():
        try:
            from report import validate_reports
            score = validate_reports()
            report_verified = prediction_verified
            report_detail = f"분석·재점검 보고서 본문과 {score['selected']}대 명단 대조 일치 / 검수 기록 있음"
        except Exception as exc:
            report_detail = f"보고서 검증 실패: {type(exc).__name__}: {exc}"
    for index, (name, detail, files) in enumerate(future, start=3):
        if index == 3 and policy_complete:
            stages.append((name, "완료", f"처리 규칙 {processing_count}줄 / 탐색 근거와 보고서 반영 확인 (구현은 4단계)"))
            continue
        if index == 4 and feature_verified:
            stages.append((name, "완료", feature_detail))
            continue
        if index == 5 and training_verified:
            stages.append((name, "완료", training_detail))
            continue
        if index == 6 and threshold_verified:
            stages.append((name, "완료", threshold_detail))
            continue
        if index == 7 and prediction_verified:
            stages.append((name, "완료", prediction_detail))
            continue
        if index == 8 and report_verified:
            stages.append((name, "완료", report_detail))
            continue
        found = any((ROOT / path).is_file() for path in files) or (index == 3 and has_processing)
        if index == 4 and feature_detail:
            detail = feature_detail
        if index == 5 and training_detail:
            detail = training_detail
        if index == 6 and threshold_detail:
            detail = threshold_detail
        if index == 7 and prediction_detail:
            detail = prediction_detail
        if index == 8 and report_detail:
            detail = report_detail
        stages.append((name, "검토 필요" if found else "대기", detail +
                       (" / 산출물 발견: 내용 검증 후 완료 판정 필요" if found else "")))
    print("\n" + "─" * 64)
    for number, (name, state, detail) in enumerate(stages, start=1):
        print(f"[{state}] {number}단계 · {name}\n       {detail}")
    complete = sum(state == "완료" for _, state, _ in stages)
    print("─" * 64)
    print(f"진행률: [{'■' * complete}{'□' * (8-complete)}] {complete}/8단계 ({complete/8:.0%})")
    print("한 줄 요약: " + "  ".join(
        f"{i}{'O' if state == '완료' else '.' if state == '대기' else '~'}"
        for i, (_, state, _) in enumerate(stages, start=1)))
    next_stage = next((f"{i}단계 · {name}" for i, (name, state, _) in enumerate(stages, start=1)
                       if state != "완료"), "모두 완료")
    print(f"다음 단계: {next_stage}")
    print("※ 완료는 해당 단계의 확인 범위 기준입니다. EDA 수치의 의미는 직접 검토해야 하며,")
    print("   3단계는 방침 작성 확인이며 보정 코드 검증이 아닙니다.")
    print("   4단계는 실제 피처를 재계산해 원본 직접 집계와 비교합니다.")
    print("   5단계는 저장 모델의 검증 예측과 지표를 재계산합니다. 2025년 시험 성능은 아닙니다.")
    print("   6단계는 비용별 최적 기준과 건수를 재계산해 보고서와 대조합니다.")
    print("   7단계는 예측을 재계산하고 시험 행 순서와 최신 드론 20대 선정을 검사합니다.")
    print("   8단계는 보고서 본문·표를 현재 근거와 대조합니다. 업무 가정·최종 승인은 담당자 확인이 필요합니다.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
