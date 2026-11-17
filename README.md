# pdm-drone-lab

교육용 가상 드론 정비 데이터로 정비 후 30일 내 재고장을 예측하고,
2026년 1월 재점검할 드론 20대를 선정하는 실습 프로젝트다.

VS Code와 AI 에이전트는 `C:\workspace\pdm-drone-lab` 폴더를 연다.

## 구성

- `AGENTS.md`: 프로젝트 공통 작업 규칙. `CLAUDE.md`와 `GEMINI.md`도 이를 참조한다.
- `.agents/skills/`: 분석·학습·보고 절차를 작성할 위치.
- `.claude/skills/`: 필요한 단계에서 스킬을 복사할 위치.
- `data/`: 원본 CSV 3개와 실습 안내 `README.txt`.
- `examples/`: 비교·검토용 예제 코드 위치.
- `scripts/`: 데이터 탐색·학습·예측·검증 코드 위치.
- `artifacts/`: 학습 모델과 모델 정보 저장 위치.
- `reports/`: 분석·검증 기록·시험 예측·재점검 계획 저장 위치.

`pdm_lib.py`는 데이터 처리·시간 피처를, `app.py`는 Streamlit 대시보드를 제공한다.
`scripts/check.py`는 1~8단계의 원본·코드 지문, 피처, 검증 예측·지표,
비용별 기준, 시험 제출 형식과 보고서 숫자를 재계산·대조한다.

## 시작 (PowerShell)

```powershell
cd C:\workspace\pdm-drone-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check.py
```

가상환경이 이미 있으면 생성은 생략한다. Python 경로가 헷갈리면
`.\.venv\Scripts\python.exe scripts/check.py`처럼 명시해서 실행한다.

## 앱과 결과 재현

```powershell
python -m streamlit run app.py
```

앱의 보고서·제출 파일 탭에서 분석 보고서와 재점검 계획을 Word로 내려받을 수 있다.
사이드바 설정은 화면 비교용이며 저장된 제출 파일은 변경하지 않는다.

원본이나 처리·학습 규칙을 바꾸면 아래 순서로 다시 생성한다.

```powershell
python scripts/verify_features.py
python scripts/train.py
python scripts/threshold_whatif.py
python scripts/predict.py
python scripts/report.py
python scripts/eda.py
python scripts/check.py
```

## 제출 파일

- `reports/test_predictions_2025.csv`: 2025년 시험 원본 1,241행에 대응하는 예측.
- `reports/analysis_report.md`: 데이터 처리 근거·검증 성능·비용 비교를 포함한 분석 보고서.
- `reports/2026-01_재점검계획.md`: 드론별 최신 기록 기준 재점검 20대 명단과 근거.

검증 기간은 2024년 7~11월이며 2025년 시험 성능과 다르다.
999 및 비용 차수 오류 보정은 가정이며, 데이터 제공자의 확인 전에는 확정 원인으로 해석하지 않는다.
파일 생성과 검수를 완료한 것이 강사 제출이나 업무 담당자 승인을 의미하지 않는다.

## 데이터

| 파일 | 기간 | 건수 | 정답 |
|---|---|---:|---|
| `data/drone_train.csv` | 2023~2024년 | 2,346 | 있음 |
| `data/drone_test.csv` | 2025년 | 1,241 | 없음 |
| `data/drone_this_month.csv` | 2026년 1월 | 106 | 없음 |

2025년 시험 성능은 강사가 채점한다. 로컬 성능 평가는 정답이 있는
학습 데이터 안에서 기간을 분리해 수행한다. 자세한 과제는 `data/README.txt`를 따른다.
