"""실행 근거를 검증한 뒤 제출 보고서를 생성하고 숫자와 명단을 대조한다."""
import hashlib
import json
import csv
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
import pandas as pd
from dashboard_analysis import context, concentration, capacity_comparison, monthly_history, month_changes, quality


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
            "- 변경 요약: 검증된 순위·확률로 재점검 후보 선정안을 작성하고 미선정 위험 드론 처리 방안과 구성 집계를 추가했다.",
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
    # 지휘관 보고 본문은 판단·건의부터 제시하고 기술 재현 정보와 명단은 붙임으로 둔다.
    with (ART / "backtest_validation.csv").open(encoding="utf-8-sig", newline="") as handle:
        monthly = list(csv.DictReader(handle))
    total_picks = sum(int(row["picked"]) for row in monthly)
    model_hits = int(card["monthly_hits"]["model"])
    old_hits = int(card["monthly_hits"]["oldest"])
    summary_analysis = [
        "# 사단장 보고용 — 드론 예측정비 방법 검증 결과 및 시범 검증 건의 (안)", "",
        "**교육용 가상 데이터 기반 실습 결과. 실제 부대의 운용 성과·점검 명령·예산 집행계획이 아님.**", "",
        f"- 작성일: {date} (Asia/Seoul). 분석 대상: 과거 2023~2025년 및 실습 시나리오 2026년 1월.",
        "- 작성 상태: AI 초안. 보고자·검토자·결재자는 실제 제출 시 지정 필요.",
        "- 변경 요약: 사단장 판단에 필요한 결론·성과 범위·적용 조건·건의를 본문으로 재구성함.",
        "", "## 1. 보고 요지",
        f"교육용 정비 기록으로 재점검 우선순위 선정 방법을 검증하였음. 검증 {len(monthly)}개월 동안 매월 20대씩 총 {total_picks}회 선정한 기록 중, 이후 30일 내 재고장이 발생한 기록은 모델 {model_hits}건·경과연수 순 {old_hits}건으로 모델이 {model_hits-old_hits}건 많았음.",
        "이는 과거 재고장 기록을 포함한 정도를 비교한 결과이며, 고장 예방 건수·실제 현장 탐지 실적·비용 절감 실적을 뜻하지 않음. 반복 선정된 같은 드론이 포함될 수 있음.",
        "", "## 2. 핵심 결과와 판단 범위", "",
        "| 구분 | 확인 결과 | 판단 범위 |", "|---|---|---|",
        f"| 과거 검증 | 모델 {model_hits}건 / 오래된 순 {old_hits}건 / 무작위 기대 {card['monthly_hits']['random_expected']:.2f}건 | 2024년 7~11월, 월별 최신 정비 기록 기준 |",
        f"| 우선순위 품질 지표(AP) | {card['validation']['pr_auc_ap']:.4f}, 기준선 {card['validation']['base_rate']:.4f} | 탐색적 내부 검증이며 독립 시험 아님 |",
        f"| 실습 재점검 후보 | {score['devices']}대 중 {score['selected']}대 | 2026년 1월 가상 데이터의 선정안 |",
        f"| 기준 이상 미선정 | {len(deferred)}대 | 실제 자원·일정과 후속 조치 미확정 |",
        "| 2025년 시험 성능 | 미확인 | 강사가 정답을 보유하며 채점 전 |",
        "", "## 3. 적용 전 확인사항",
        "- 원본의 운용중단 999 및 비용 극단값 보정 원인을 데이터 제공자에게 확인할 필요가 있음. 보정 전후 월 적중 합계는 같았으며 입력 오류가 입증된 것은 아님.",
        "- 실제 정비 이력의 정의·입력 품질·정비 완료 시점에 확보되는 정보를 확인하고 별도 기간으로 검증할 필요가 있음.",
        "- 실제 비용과 월 점검 가능 대수는 별도 확인 대상임. 실습의 300만원·60만원 가정은 예산 요구나 확정 단가가 아님.",
        "- 모델 예측 확률과 실제 발생률의 일치는 검증하지 않았으므로 정확한 고장 확률이나 안전 판정으로 사용하지 않음.",
        "", "## 4. 건의사항 및 확인 절차",
        "교육용 결과를 근거로 실제 적용을 확정하기보다, 다음 조건을 확인하는 별도 시범 검증의 추진 여부를 검토해 주시기 바람.",
        "", "| 검토 사항 | 제안 역할 | 완료 기준 | 담당·기한 |", "|---|---|---|---|",
        "| 데이터 보정 가정 확인 | 데이터 제공자·정비 담당 | 코드·단위·결측 의미 확인 | 지정·협의 필요 |",
        "| 실제 이력 기반 별도 검증 | 분석 담당·정비 담당 | 기간 분리 및 현행 선정 기준과 비교 | 지정·협의 필요 |",
        "| 현장 적용·확대 여부 판단 | 지휘관·정비 책임자 | 추가 검증 결과, 자원, 미선정 위험 처리 방안 검토 | 별도 판단 필요 |",
        "- 검토 승인·예산 승인·시범 운용 승인이 이루어진 것으로 표현하지 않음.",
        "", "## 붙임. 데이터·분석·검증 상세 근거",
    ]
    # 기존 상세 내용을 보존하되 붙임의 제목 계층을 낮춘다.
    detail_start = analysis.index("## 1. 목적과 데이터")
    analysis_details = "\n".join(analysis[detail_start:]).splitlines()
    analysis_details = ["#" + line if line.startswith("## ") else line for line in analysis_details]
    analysis = summary_analysis + commands + analysis_details
    # 모델 한계는 보고서 끝에 원문 그대로 유지한다.
    analysis = analysis[:-len(limits)] + limits

    plan_summary = [
        "# 대대장 보고용 — 2026년 1월 드론 재점검 후보 선정안",
        "", "**교육용 가상 데이터의 과거 시나리오. 현재 운용 중인 실제 드론의 점검 지시가 아님.**", "",
        f"- 작성일: {date} (Asia/Seoul). 대상 시점: 실습 데이터 2026년 1월.",
        "- 작성 상태: AI 초안. 보고자·검토자·최종 승인자 지정 필요.",
        "- 변경 요약: 점검 후보·용량 초과·후속 조치와 건의를 앞에 두고 상세 명단·기술 근거를 붙임으로 이동함.",
        "", "## 1. 보고 요지",
        f"실습 데이터의 정비 기록 {score['records']}건, 드론 {score['devices']}대에서 우선 재점검 후보 {score['selected']}대를 제안함. 위험 표시 기준 {score['threshold']:.0%} 이상 {score['high_risk']}대 중 {len(deferred)}대는 월 점검 가정 20대 범위에 포함되지 않음.",
        "명단은 모델의 우선순위 제안이며 담당자의 현재 상태 확인과 승인 전에는 확정 대상이나 비행 가능 여부 판정으로 사용하지 않음.",
        "", "## 2. 판단·건의사항",
        f"- 실습상 상위 {score['selected']}대의 이력·현재 상태를 대조하는 검토 절차를 제안함. 실제 적용 여부는 별도 판단 사항임.",
        f"- 미선정 위험 {len(deferred)}대는 추가 점검 자원 검토 또는 우선순위 재검토 방안을 협의하도록 제안함. 자동으로 다음 달로 이월하지 않음.",
        "- 담당자·검토 완료 기한·확인 결과 보고 시점을 지정할 필요가 있음. 현재 미정이며 아래 표는 조치 제안임.",
        "", "## 3. 자원과 후속 조치 (안)",
        "| 대상 | 제안 조치 | 제안 역할 | 기한·결정 상태 |", "|---|---|---|---|",
        f"| 후보 {score['selected']}대 | 이력·현 상태 확인 후 재점검 순서 검토 | 정비 담당 | 지정·협의 필요 |",
        f"| 미선정 위험 {len(deferred)}대 | 현 상태·중요도 재검토, 추가 자원 또는 순서 조정안 보고 | 정비 책임자 | 지정·협의 필요 |",
        "| 결과 확인 | 실제 재고장·정비 결과 기록, 선정 방식과 비교 | 정비·분석 담당 | 시행 여부 판단 후 설정 |",
        f"- 실습 단가 60만원을 {score['selected']}대에 단순 적용한 점검비 가정은 {score['selected']*60:,}만원임. 실제 예산 요구액이 아니며 인력·시간·부품·실제 단가와 가용 자원은 미산정임.",
        "", "## 4. 해석 시 주의사항",
        "- 표의 확률은 모델 예측값이며 실제 고장 발생률과 일치하는지는 미검증임. 높은 순위만으로 운용 중지·비행 허용·부품 교체를 결정하지 않음.",
        f"- 검증 {len(monthly)}개월의 총 {total_picks}회 선정 중 모델 {model_hits}건·오래된 순 {old_hits}건의 재고장 기록을 포함했음. 실제 예방 효과는 미확인임.",
        "- 상위 위험 요인은 모델 예측의 기여 항목이며 고장 원인이나 조치 효과를 확정한 것이 아님.",
        "", "## 붙임. 재점검 후보 명단·위험 요인·미선정 목록", "",
        "표의 값은 scripts/predict.py의 reports/score_2026-01.json을 인용함. 후보 명단은 담당자 검토 전 선정안임.",
    ]
    plan_detail_start = plan.index("## 1. 요약")
    plan_details = ["#" + line if line.startswith("## ") else line for line in plan[plan_detail_start:]]
    plan = plan_summary + commands + plan_details
    plan = plan[:-len(limits)] + limits
    ranked = pd.DataFrame(score["rows"])
    ranked["재점검"] = ranked["selected"]
    ranked["위험 높음"] = ranked["risk_high"]
    raw_month, history = context(score)
    capacity_rows = capacity_comparison(ranked, score["threshold"], score["capacity"])
    quality_result = quality(raw_month, history)
    changes = month_changes(ranked, monthly_history(history, score["period"]), score["period"])

    def markdown_table(frame):
        rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            cells = ["미확인" if pd.isna(v) else f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    dashboard = ["# 대시보드 추가 분석 결과", "", f"- 갱신일: {date} (Asia/Seoul)",
                 "- 대상: 기본 교육용 2026년 1월 데이터. 업로드 자료는 이 저장 보고서에 포함하지 않는다.",
                 "- 근거: scripts/dashboard_analysis.py, data/drone_*.csv, reports/score_2026-01.json, artifacts/drone_model.joblib.",
                 "- 재현: python scripts/report.py. 화면: python -m streamlit run app.py.",
                 "", "## 1. 정비 이력 타임라인",
                 "정비일·정비 구분·고장 유형·정비 성공·최근 90일 고장 수·예방정비 후 비행시간을 드론별로 조회한다. 대상 월 이후의 기록을 제외한다.",
                 "", "## 2. 유형·부대별 위험 집중",
                 "위험 비율은 해당 집단의 기준 이상 드론 수를 정비 드론 수로 나눈 값으로 실제 재고장률이 아니다."]
    for field, label in [("equipment_type", "유형"), ("unit_code", "부대")]:
        dashboard += [markdown_table(concentration(ranked, field).rename(columns={field: label}))]
    dashboard += ["", "## 3. 점검 용량 비교", markdown_table(capacity_rows),
                  "동일 위험 순위를 유지한 비교이며 실제 고장 예방·적중·절감액이 아니다. 위험 드론이 없으면 포함률은 미확인이다.",
                  "", "## 4. 월별 변화 (현재 최종 모델 재적용)",
                  "전월 기록이 있는 드론의 예측값 상승 순으로 최대 10대를 표시한다. 전월이 없으면 비교에서 제외한다. 당시 예측 보관값이나 독립 과거 성능이 아니다.",
                  markdown_table(changes.loc[changes.previous_prob.notna()].head(10).rename(columns={
                      "equipment_id": "드론", "prob": "현재 예측값", "previous_prob": "전월 예측값",
                      "previous_fail_count_90d": "전월 90일 고장 수", "fail_count_90d": "현재 90일 고장 수"})),
                  "", "## 5. 원본 입력 품질",
                  f"- 원본 결측 셀 {int(raw_month.isna().sum().sum())}개, 드론·정비일 중복 {quality_result['duplicates']}건, 이전 이력 없는 드론 {len(quality_result['no_history'])}대.",
                  "- 예방정비 고장 정보의 정상 결측을 포함한 값이며 모두 오류라는 의미가 아니다.",
                  markdown_table(quality_result["ranges"]), markdown_table(quality_result["unknown"]),
                  "- 학습 범위 밖 값은 오류 확정이 아니며 노후화·누적 비행시간 증가 등으로도 발생한다. 새 입력은 기존 모델로 조회하며 자동 재학습하지 않는다."] + limits
    analysis[-len(limits):-len(limits)] = ["", "## 추가 조회 분석", "위험 집중·점검 용량·월별 변화·입력 품질의 상세 결과는 reports/dashboard_analysis.md를 참조한다. 기본 월 자료의 탐색 결과이며 현장 효과는 미검증이다."]
    return {"analysis_report.md": "\n".join(analysis) + "\n", "2026-01_재점검계획.md": "\n".join(plan) + "\n",
            "dashboard_analysis.md": "\n".join(dashboard) + "\n"}, score


def dependencies():
    paths = [ROOT / "scripts/report.py", ROOT / ".agents/skills/pdm-risk/SKILL.md",
             ROOT / "reports/model_validation.md", ROOT / "reports/threshold_whatif.md",
             ART / "prediction_manifest.json", ROOT / "scripts/dashboard_analysis.py"]
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
