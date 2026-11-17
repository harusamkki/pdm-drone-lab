# 드론 예측정비 데이터 분석 보고서

- 마지막 갱신일: 2026-11-17 (Asia/Seoul)
- 변경 요약: 데이터 탐색·처리·피처 검증·기간 검증·비용 비교·시험 예측을 제출용으로 정리하고 수치를 대조했다.
- 작성: AI 초안. 업무 가정 확인·최종 승인: 정비 담당자 확인 필요.
- 적용 스킬: .agents/skills/pdm-train/SKILL.md, .agents/skills/pdm-risk/SKILL.md.
- 재현 명령: `.\.venv\Scripts\python.exe scripts/report.py`.
- 재계산 순서: scripts/train.py → scripts/threshold_whatif.py → scripts/predict.py → scripts/report.py → scripts/check.py.

## 1. 목적과 데이터
정비 직후 알 수 있는 정보로 30일 안 재고장을 예측하고 실습 대상 월 2026년 1월의 재점검 드론 20대를 제안한다.
- 근거: data/README.txt, data/drone_train.csv, data/drone_test.csv, data/drone_this_month.csv. 상세 탐색: reports/eda.md.

| 구분 | 기간 | 정비 기록 | 정답 |
|---|---|---:|---|
| 학습 원본 | 2023~2024년 | 2346 | 있음 |
| 시험 제출 | 2025년 | 1241 | 없음 |
| 재점검 | 2026년 1월 | 106 | 없음 |
- 학습 재고장은 525건(22.38%). 정비 기록 기준이며 독립 드론 표본 비율이 아니다.

## 2. 데이터 처리와 근거
- 근거: AGENTS.md, pdm_lib.py, reports/eda.md, reports/feature_verification.md.
- 학습의 운용중단 999인 4건은 미상값으로 가정해 결측 처리했다.
- 학습 비용 15,000·16,200·16,800·18,300·19,000의 각 1건은 100으로 나눴다. 보정 후 학습 비용 최대는 478만원이다. 차수 오류는 확정 사실이 아닌 가정이다.
- 예방정비의 고장코드·유형 결측은 N/A로 채웠다. 각 컬럼 기준 학습 1,707건·시험 876건·이번 달 62건이다.
- 경과일 결측 150건은 그대로 두고 최초 기록 표시를 추가했다. 제공 이력의 첫 기록이며 생애 첫 정비로 단정하지 않는다.
- 행은 삭제하지 않았고 원본을 수정하지 않았다. 최근 90일 고장 횟수·최근 고장 후 경과일·예방정비 후 비행시간은 정비일까지의 이력으로 계산했다.
- 시험·이번 달 피처는 이전 기간 이력을 이어 계산한다. 미래 날짜나 정답을 이용하지 않는다. 전 행 검산과 표본 검산을 통과했다.

## 3. 입력 컬럼과 제외 이유
- 근거: scripts/train.py의 FEATURES, artifacts/model_card.json.
- 입력: `equipment_type`, `maintenance_type`, `fault_category`, `equipment_age_years`, `severity`, `downtime_hours`, `repair_duration_hours`, `parts_replaced_count`, `maintenance_cost`, `maintenance_success`, `cumulative_operating_hours`, `days_since_last_maintenance`, `is_first_maintenance`, `fail_count_90d`, `days_since_last_failure`, `hours_since_last_pm`.
| 제외 컬럼 | 이유 |
|---|---|
| availability_rate | 예측 시점에 모르는 월 전체 가동률 |
| equipment_id, unit_code | 식별자이며 순위 표시·집계에만 사용 |
| maintenance_date | 기간 분리·이력 계산에 사용하고 입력 원값은 제외 |
| fault_code | fault_category로 대체 |
| failure_within_30days | 정답이므로 입력 금지 |

## 4. 모델 학습과 기간 검증
다음 내용은 reports/model_validation.md의 실행 결과이며 2025년 시험 성능이 아니다.
### 학습 결과

- 갱신일: 2026-11-17 (Asia/Seoul)
- 근거: artifacts/model_card.json, artifacts/validation_predictions.csv, artifacts/backtest_validation.csv.
- 재현 명령: `.\.venv\Scripts\python.exe scripts/train.py`. 적용 스킬: .agents/skills/pdm-train/SKILL.md.
- 초기 학습 1611건, 마지막 정비일 2024-05-31. 정비 후 30일이 검증 시작 전에 끝나는 기록만 사용했다.
- 검증: 2024년 7~11월 515건. 최종 모델은 2025년 시작 전에 정답이 확정되는 2223건으로 학습했다.

## 검증 성능 (정비 기록 단위)

| 지표 | 결과 | 기준 |
|---|---:|---|
| PR-AUC (AP) | 0.7549 | 재고장 비율 0.2524 |
| ROC-AUC | 0.8742 | 무작위 0.5 |
| 정확도 | 82.72% | 전체 음성 74.76% |
| 정밀도 | 61.99% | 기준 0.26 |
| 재현율 | 81.54% | 같은 검증에서 고른 기준 |

## 월 20대 적중 비교 (드론별 월 최신 기록)

| 월 | 드론 수 | 선정 | 모델 | 오래된 순 | 무작위 기대 |
|---|---:|---:|---:|---:|---:|
| 2024-07 | 77 | 20 | 6 | 4 | 2.34 |
| 2024-08 | 87 | 20 | 4 | 2 | 1.61 |
| 2024-09 | 79 | 20 | 10 | 6 | 3.54 |
| 2024-10 | 89 | 20 | 5 | 2 | 1.35 |
| 2024-11 | 78 | 20 | 5 | 2 | 2.05 |
| 합계 | — | 100 | 30 | 16 | 10.89 |

## 비용 기준과 처리 가정 민감도
- 놓친 고장 300만원·점검 60만원에서 검증 비용 최소 기준은 0.26다. 기록별 비용이며 월 20대 제약은 별도다.
- 보정 적용: AP 0.7549, 월 적중 합계 30건.
- 극단값 원값 유지: AP 0.7522, 월 적중 합계 30건.
- 한 번의 검증 비교이며 보정 원인이 사실임을 증명하지 않는다. 결과에 맞춰 주 모델이나 규칙을 바꾸지 않았다.



## 5. 운영 비용과 임계값
- 근거: reports/threshold_whatif.md, artifacts/validation_predictions.csv. 재점검 단가 60만원 고정.
| 놓친 재고장 비용(만원/건) | 기준 | 기준 이상 건수 | 놓친 재고장 수 | 최소 총비용(만원) |
|---|---:|---:|---:|---:|
| 150 | 0.30 | 156 | 28 | 13,560 |
| 300 | 0.26 | 171 | 24 | 17,460 |
| 600 | 0.10 | 287 | 11 | 23,820 |
- 놓치는 비용이 커질수록 기준이 낮아지고 점검 대상 기록은 늘었다. 검증 기록 전체 비용 최적화이며 월 20대 제약은 반영하지 않는다.
- 실제 저장 기준은 0.26로 유지했다. 600만원 비교는 시나리오이며 운영 기준을 자동 변경하지 않았다.

## 6. 시험 제출과 이번 달 선정
- 시험 제출: reports/test_predictions_2025.csv. 1,241행 모두 원본과 같은 순서로 대응하며 열은 equipment_id, maintenance_date, prob다. 확률 유한값·0~1 범위·모델 재계산 일치를 확인했다.
- 2025년 시험 정답이 없으므로 시험 성능은 미확인이다. 제출 파일을 만든 것이 강사에게 제출한 것은 아니다.
- 이번 달 81대 중 20대를 선정했다. 기준 26% 이상은 29대이며 미선정 위험 드론은 9대다.
- 상세 계획: reports/2026-01_재점검계획.md. 점수 근거: reports/score_2026-01.json. CSV 명단: reports/reinspection_2026-01.csv.

## 모델 한계·최종 판단
- 교육용 가상 데이터이며 실제 운용에 그대로 적용하지 않는다.
- 검증 기간에서 임계값도 선택했으므로 독립 시험 성능이 아니다. 2025년 시험 정답은 제공되지 않았다.
- 극단값 보정 원인은 확정되지 않은 가정이며 같은 드론의 반복 기록이 포함된다.
- 월별 최신 정비 기록의 재고장 결과로 평가했으며 실제 점검 효과를 입증하지 않는다.
- 확률 보정은 별도로 검증하지 않았으며 최종 판단은 정비 담당자가 한다.
