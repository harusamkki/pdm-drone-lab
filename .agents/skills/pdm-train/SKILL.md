---
name: pdm-train
description: 드론 정비 이력으로 30일 내 재고장 모델을 학습하고 기간 검증, 기준선 비교, 임계값 선택과 모델 저장을 수행한다. 드론 모델 학습·재학습·검증 요청에 사용한다.
---

# 드론 모델 학습·검증

`scripts/train.py`를 실행해 `artifacts/`에 모델·검증 예측·평가 정보를, `reports/model_validation.md`에 실행 결과를 저장한다.

1. `pdm_lib.load_records('train')` → `clean` → `add_time_features`로 학습 CSV만 준비한다. `FEATURES`는 `scripts/train.py`에 고정하고 제외 대상은 AGENTS.md를 따른다.
2. 검증은 2024-07-01 이상 2024-12-01 미만으로 한다. 초기 학습은 정비일+30일 < 2024-07-01로 제한한다. 같은 드론이 두 기간에 나타나는 것은 기존 드론의 미래 정비를 예측하는 목적에 부합한다.
3. 고정된 `make_model()`로 학습한다. 원핫 인코더는 학습에만 적합하며 수치 결측은 LightGBM에 맡긴다. 시드 42를 사용한다.
4. 검증 전체 정비 기록의 Average Precision·ROC-AUC·정답 비율과 분류 지표를 계산한다. 임계값은 0.05~0.94(0.01 간격) 중 놓친 고장×300+기준 이상×60이 최소인 값으로 선택한다. 동률은 낮은 기준을 택한다.
5. 월별 최신 기록으로 드론 중복을 제거하고 확률 상위 20대 적중 수를 센다. 동일 모집단의 오래된 순·무작위 기대값과 비교한다. 정비 기록 단위 AP와 드론 단위 월 적중 수를 구분한다.
6. 고정한 동일 모델에서 극단값만 원값으로 유지한 대안을 한 번 비교해 처리 가정의 민감도를 기록한다. 주 모델이나 규칙을 결과에 맞춰 자동 교체하지 않는다.
7. 최종 모델은 학습 CSV 중 정비일+30일 < 2025-01-01인 기록으로 학습한다. `artifacts/validation_model.joblib`, `artifacts/drone_model.joblib`, `artifacts/validation_predictions.csv`, `artifacts/backtest_validation.csv`, `artifacts/model_card.json`을 저장한다.
8. `python scripts/train.py` 후 `python scripts/check.py`로 결과를 확인한다. 보고서에 모델 정보의 한계를 기록한다.

- 2025년 정답은 없으므로 성능을 계산하지 않는다. 시험 CSV 예측과 이번 달 20대 선정은 다음 단계 요청에서 수행한다.
- 검증 기간에서 임계값도 선택했음을 밝힌다. 아직 독립적인 정답 있는 시험 평가가 아니다.
- 숫자는 실행 결과만 인용한다. 이전 장비 실습의 점수·임계값을 기대값으로 사용하지 않는다.
