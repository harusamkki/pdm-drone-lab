"""실행 근거를 검증한 뒤 제출 보고서를 생성하고 숫자와 명단을 대조한다."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pdm_lib import load_records
from train import ART, validate_saved_training
from predict import REPORTS, validate_outputs
from threshold_whatif import validate_report, result_table


def render(date):
    card = validate_saved_training()
    score = validate_outputs()
    cost_rows = validate_report()
    train = load_records("train")
    chosen = [r for r in score["rows"] if r["selected"]]
    deferred = [r for r in score["rows"] if r["risk_high"] and not r["selected"]]
    assert len(chosen) == 20 and len({r["equipment_id"] for r in chosen}) == 20
    assert len(deferred) == score["high_risk_unselected"]
    commands = ["- 적용 스킬: .agents/skills/pdm-train/SKILL.md, .agents/skills/pdm-risk/SKILL.md.",
                "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/report.py`.",
                "- 재계산 순서: scripts/train.py → scripts/threshold_whatif.py → scripts/predict.py → scripts/report.py → scripts/check.py."]
    limits = ["", "## 모델 한계·최종 판단"] + ["- " + x for x in card["limits"]]
    analysis = ["# 드론 예측정비 데이터 분석 보고서", "",
                f"- 마지막 갱신일: {date} (Asia/Seoul)",
                "- 변경 요약: 데이터 탐색·처리·피처 검증·기간 검증·비용 비교·시험 예측을 제출용으로 정리하고 수치를 대조했다.",
                "- 작성: AI 초안. 업무 가정 확인·최종 승인: 정비 담당자 확인 필요."] + commands
    analysis += ["", "## 1. 목적과 데이터",
                 "정비 직후 알 수 있는 정보로 30일 안 재고장을 예측하고 실습 대상 월 2026년 1월의 재점검 드론 20대를 제안한다.",
                 "- 근거: data/README.txt, data/drone_train.csv, data/drone_test.csv, data/drone_this_month.csv. 상세 탐색: reports/eda.md.",
                 "", "| 구분 | 기간 | 정비 기록 | 정답 |", "|---|---|---:|---|",
                 f"| 학습 원본 | 2023~2024년 | {len(train)} | 있음 |",
                 f"| 시험 제출 | 2025년 | {len(load_records('test'))} | 없음 |",
                 f"| 재점검 | 2026년 1월 | {score['records']} | 없음 |",
                 f"- 학습 재고장은 {int(train.failure_within_30days.sum())}건({train.failure_within_30days.mean():.2%}). 정비 기록 기준이며 독립 드론 표본 비율이 아니다.",
                 "", "## 2. 데이터 처리와 근거",
                 "- 근거: AGENTS.md, pdm_lib.py, reports/eda.md, reports/feature_verification.md.",
                 "- 학습의 운용중단 999인 4건은 미상값으로 가정해 결측 처리했다.",
                 "- 학습 비용 15,000·16,200·16,800·18,300·19,000의 각 1건은 100으로 나눴다. 보정 후 학습 비용 최대는 478만원이다. 차수 오류는 확정 사실이 아닌 가정이다.",
                 "- 예방정비의 고장코드·유형 결측은 N/A로 채웠다. 각 컬럼 기준 학습 1,707건·시험 876건·이번 달 62건이다.",
                 "- 경과일 결측 150건은 그대로 두고 최초 기록 표시를 추가했다. 제공 이력의 첫 기록이며 생애 첫 정비로 단정하지 않는다.",
                 "- 행은 삭제하지 않았고 원본을 수정하지 않았다. 최근 90일 고장 횟수·최근 고장 후 경과일·예방정비 후 비행시간은 정비일까지의 이력으로 계산했다.",
                 "- 시험·이번 달 피처는 이전 기간 이력을 이어 계산한다. 미래 날짜나 정답을 이용하지 않는다. 전 행 검산과 표본 검산을 통과했다.",
                 "", "## 3. 입력 컬럼과 제외 이유",
                 "- 근거: scripts/train.py의 FEATURES, artifacts/model_card.json.",
                 "- 입력: " + ", ".join(f"`{x}`" for x in card["features"]) + ".",
                 "| 제외 컬럼 | 이유 |", "|---|---|",
                 "| availability_rate | 예측 시점에 모르는 월 전체 가동률 |",
                 "| equipment_id, unit_code | 식별자이며 순위 표시·집계에만 사용 |",
                 "| maintenance_date | 기간 분리·이력 계산에 사용하고 입력 원값은 제외 |",
                 "| fault_code | fault_category로 대체 |",
                 "| failure_within_30days | 정답이므로 입력 금지 |",
                 "", "## 4. 모델 학습과 기간 검증",
                 "다음 내용은 reports/model_validation.md의 실행 결과이며 2025년 시험 성능이 아니다.",
                 (REPORTS / "model_validation.md").read_text(encoding="utf-8").split("## 한계")[0].replace("# 드론 모델 학습·검증 보고서", "### 학습 결과", 1),
                 "", "## 5. 운영 비용과 임계값",
                 "- 근거: reports/threshold_whatif.md, artifacts/validation_predictions.csv. 재점검 단가 60만원 고정.",
                 result_table(cost_rows),
                 "- 놓치는 비용이 커질수록 기준이 낮아지고 점검 대상 기록은 늘었다. 검증 기록 전체 비용 최적화이며 월 20대 제약은 반영하지 않는다.",
                 f"- 실제 저장 기준은 {card['threshold']:.2f}로 유지했다. 600만원 비교는 시나리오이며 운영 기준을 자동 변경하지 않았다.",
                 "", "## 6. 시험 제출과 이번 달 선정",
                 "- 시험 제출: reports/test_predictions_2025.csv. 1,241행 모두 원본과 같은 순서로 대응하며 열은 equipment_id, maintenance_date, prob다. 확률 유한값·0~1 범위·모델 재계산 일치를 확인했다.",
                 "- 2025년 시험 정답이 없으므로 시험 성능은 미확인이다. 제출 파일을 만든 것이 강사에게 제출한 것은 아니다.",
                 f"- 이번 달 {score['devices']}대 중 {score['selected']}대를 선정했다. 기준 {score['threshold']:.0%} 이상은 {score['high_risk']}대이며 미선정 위험 드론은 {len(deferred)}대다.",
                 "- 상세 계획: reports/2026-01_재점검계획.md. 점수 근거: reports/score_2026-01.json. CSV 명단: reports/reinspection_2026-01.csv."]
    analysis += limits
    plan = ["# 2026-01 드론 재점검 계획 (안)", "",
            f"- 마지막 갱신일: {date} (Asia/Seoul)",
            "- 변경 요약: 검증된 순위·확률로 명단을 확정하고 미선정 위험 드론 처리 방안과 구성 집계를 추가했다.",
            "- 작성: AI 초안. 검토·최종 승인: 정비 담당자 확인 필요.",
            "- 근거: reports/score_2026-01.json, reports/reinspection_2026-01.csv, artifacts/model_card.json."] + commands
    plan += ["", "## 1. 요약",
             f"- 대상 월: 실습 데이터의 2026년 1월. 정비 기록 {score['records']}건, 드론 {score['devices']}대.",
             f"- 확률 상위 {score['selected']}대 선정. 위험 표시 기준 {score['threshold']:.0%}, 기준 이상 {score['high_risk']}대, 그중 미선정 {len(deferred)}대.",
             "- 드론별 가장 최근 정비 기록으로 판단하고 확률 동률은 드론 번호 순이다. 모든 확률은 해당 정비 후 30일 기준이다.",
             "", "## 2. 재점검 대상", "",
             "| 순위 | 드론 | 유형 | 부대 | 최근 정비일 | 확률 | 위험을 올린 요인 | 권고 |",
             "|---|---|---|---|---|---:|---|---|"]
    for r in chosen:
        plan.append(f"| {r['rank']} | {r['equipment_id']} | {r['equipment_type']} | {r['unit_code']} | {r['maintenance_date']} | {r['prob']:.1%} | {r['reasons']} | {r['recommendation']} |")
    plan += ["", "## 3. 선정 근거와 구성"]
    for r in chosen[:3]:
        plan.append(f"- {r['equipment_id']}: {r['prob']:.1%}. 양의 예측 기여 상위 요인은 {r['reasons']}이다.")
    plan += ["- 예측 기여는 원인이나 점검 효과를 의미하지 않는다.",
             "- 유형별: " + ", ".join(f"{k} {v}대" for k,v in Counter(r['equipment_type'] for r in chosen).most_common()) + ".",
             "- 부대별: " + ", ".join(f"{k} {v}대" for k,v in Counter(r['unit_code'] for r in chosen).most_common()) + ".",
             "- 선정 수가 많다는 이유만으로 해당 부대·유형의 고장률이 높다고 단정하지 않는다.",
             "", "## 4. 위험 기준 이상 미선정 드론",
             "담당자는 현재 상태와 임무 중요도, 추가 점검 자원을 확인해 일정을 정한다. 자동으로 다음 달로 미루지 않으며 급박한 이상 징후가 있으면 순위와 별도로 판단한다.",
             "", "| 전체 순위 | 드론 | 확률 |", "|---|---|---:|"]
    for r in deferred:
        plan.append(f"| {r['rank']} | {r['equipment_id']} | {r['prob']:.1%} |")
    plan += ["", "## 5. 모델 평가의 범위",
             f"- 2024년 7~11월 검증 AP {card['validation']['pr_auc_ap']:.4f}; 기준 재고장 비율 {card['validation']['base_rate']:.4f}.",
             f"- 같은 기간 월 20대 적중 합계: 모델 {card['monthly_hits']['model']:.0f}, 오래된 순 {card['monthly_hits']['oldest']:.0f}, 무작위 기대 {card['monthly_hits']['random_expected']:.2f}건.",
             f"- 최종 모델 학습 기록은 {card['final_train_rows']}건, 마지막 정비일 {card['final_last_date']}다. 검증 모델과 구분한다.",
             "- 2025년 시험과 2026년 1월의 정답은 없어 실제 적중과 점검 효과는 미확인이다."] + limits
    return {"analysis_report.md": "\n".join(analysis) + "\n", "2026-01_재점검계획.md": "\n".join(plan) + "\n"}, score


def dependencies():
    paths = [ROOT / "scripts/report.py", ROOT / ".agents/skills/pdm-risk/SKILL.md",
             ROOT / "reports/model_validation.md", ROOT / "reports/threshold_whatif.md",
             ART / "prediction_manifest.json"]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_reports():
    manifest = json.loads((ART / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == dependencies(), "보고서 근거 변경: report.py 재실행 필요"
    expected, score = render(manifest["date"])
    for name, text in expected.items():
        actual = (REPORTS / name).read_text(encoding="utf-8")
        assert actual == text, f"보고서 본문·표 불일치: {name}"
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((REPORTS / name).read_bytes()).hexdigest() == digest, f"검수 기록 변경: {name}"
    return score


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    reports, score = render(date)
    for name, text in reports.items():
        (REPORTS / name).write_text(text, encoding="utf-8")
    verification = ["# 제출 보고서 숫자 검수 기록", "", f"- 확인일: {date} (Asia/Seoul)",
                    "- 실행: scripts/report.py. 검수자는 AI이며 정비 담당자 승인을 대체하지 않는다.",
                    "- 데이터 처리·피처 검증은 reports/feature_verification.md 및 scripts/check.py에서 별도로 확인한다.",
                    "- 학습 근거 지문, 저장 평가 모델 예측·지표, 월별 기준선 적중 합계 재계산 통과.",
                    "- 150·300·600만원 비용별 최적 기준·대상 건수·놓친 건수 재계산 통과.",
                    "- 시험 1,241행의 장비·정비일·순서·확률 범위와 저장 모델 재계산 일치.",
                    f"- 이번 달 {score['records']}건 / {score['devices']}대 / 선정 {score['selected']}대 / 위험 높음 {score['high_risk']}대 / 미선정 위험 {score['high_risk_unselected']}대.",
                    "- 계획 명단 20대의 순위·확률·유형·부대·위험 요인과 요약·구성 집계는 score_2026-01.json에서 생성해 대조했다.",
                    "- 모델 limits를 두 보고서 끝에 그대로 포함했다. 2025년 시험 성능을 보고하지 않았다.",
                    "- 불일치: 없음. 주의: 극단값 보정 원인은 가정이며, 실제 비용 절감과 점검 효과는 미확인이다.",
                    "- 검수 범위: 보고서 본문·표를 현재 근거에서 재구성해 전체 대조한다. 표현 변경 시에도 재검토가 필요하다."]
    (REPORTS / "verification.md").write_text("\n".join(verification) + "\n", encoding="utf-8")
    files = list(reports) + ["verification.md"]
    manifest = {"date": date, "dependencies": dependencies(),
                "outputs": {name: hashlib.sha256((REPORTS / name).read_bytes()).hexdigest() for name in files}}
    (ART / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validate_reports()
    print("PASS: 분석 보고서·재점검 계획·검수 기록 생성, 20대 명단과 본문 수치 대조 통과")


if __name__ == "__main__":
    main()
