# 4단계 데이터 처리·피처 생성 검증

- 확인일: 2026-11-17 (Asia/Seoul)
- 재현 명령: `.\.venv\Scripts\python.exe scripts/verify_features.py`
- 근거: data/drone_*.csv, pdm_lib.py, AGENTS.md. 원본 SHA256 유지 확인 완료.
- 검증 통과: 원본 보존, 비용·결측 보정, 전체 행 직접 검산, 90일 경계, 같은 날짜, 정비 파일 간 이력 연결, 장비 분리, 중복 인덱스, 미래 기록·정답 변경의 영향 없음.

## 데이터별 결과

| 파일 | 행 수 | 고장 횟수 합 | 직전 고장 일수 결측 | 예방정비 후 시간 결측 |
|---|---:|---:|---:|---:|
| train | 2346 | 3427 | 772 | 1 |
| test | 1241 | 2671 | 108 | 0 |
| this_month | 106 | 155 | 7 | 0 |

## 표본 직접 검산

- 장비: DR-0051, 정비일: 2023-09-11.
- 최근 90일 범위: 2023-06-13 ~ 2023-09-11 (양 끝 포함).
- 원본 기록에서 직접 센 고장 정비: 1건 / 구현한 값: 1건. 일치한다.

| 정비일 | 정비 유형 | 고장 횟수 포함 |
|---|---|---|
| 2023-07-02 | 예방정비 | 제외 |
| 2023-07-10 | 사후정비 | 포함 |
| 2023-09-11 | 예방정비 | 제외 |

## 검증 시점의 파일 지문

- pdm_lib.py: 22fb39826c7e31370163cbcdef945ea7e7e4aadadfc93c93a01f62f5c3faccf7
- AGENTS.md: e5f55ccb50f470378d72906600718276420b579e33053c901ec701346f8afc71
- scripts/verify_features.py: c4da47fb10aa03f49cdf7433fe2a9d0d317b3d6cd34f887e5db7d15d6cc63bd8
- data/drone_test.csv: 5fa789dd8111ce5c3303e9a5c9f0694aec409a1c3a504574cc2ab2386ece7a7e
- data/drone_this_month.csv: 379631f03af5139d8a0c5c7278d812b7892da022a4d7650488eafe5ff564e346
- data/drone_train.csv: 26c4d3711fd5c91dc183815af83ba5af6107b6470ae2d3660da2a50586136121
