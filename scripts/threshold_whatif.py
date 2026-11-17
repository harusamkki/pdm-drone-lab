"""검증 예측의 비용 가정을 비교한다. 모델·학습 코드·운영 기준은 바꾸지 않는다."""
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ART, pick_threshold, validate_saved_training

REPORT = ROOT / "reports/threshold_whatif.md"
PREDICTION = "놓친 재고장 비용이 커지면 임계값은 낮아지는 방향, 작아지면 높아지는 방향을 예상한다. 이산 탐색에서는 같은 기준이 유지될 수도 있다."


def calculate():
    card = validate_saved_training()
    valid = pd.read_csv(ART / "validation_predictions.csv")
    y, prob = valid.failure_within_30days.to_numpy(), valid.prob.to_numpy()
    thresholds = np.round(np.arange(.05, .95, .01), 2)
    rows = []
    for cost in [150, 300, 600]:
        threshold = pick_threshold(y, prob, cost)
        flagged = int((prob >= threshold).sum())
        missed = int(((prob < threshold) & (y == 1)).sum())
        total = missed * cost + flagged * 60
        # 다른 기준 모두와 비교해 선택 비용이 최소인지 확인한다.
        candidates = [(int(((prob < t) & (y == 1)).sum()) * cost + int((prob >= t).sum()) * 60, t)
                      for t in thresholds]
        assert (total, threshold) == min(candidates)
        rows.append((cost, threshold, flagged, missed, total))
    assert rows[1][1] == card["threshold"], "300만원 기준 재현 실패"
    return valid, rows


def result_table(rows):
    return "\n".join(["| 놓친 재고장 비용(만원/건) | 기준 | 기준 이상 건수 | 놓친 재고장 수 | 최소 총비용(만원) |",
                      "|---|---:|---:|---:|---:|"] +
                     [f"| {cost} | {threshold:.2f} | {flagged} | {missed} | {total:,} |"
                      for cost, threshold, flagged, missed, total in rows])


def fingerprints():
    paths = [ART / "validation_predictions.csv", ART / "model_card.json", ROOT / "scripts/threshold_whatif.py"]
    return [f"- {p.relative_to(ROOT).as_posix()}: {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in paths]


def validate_report():
    valid, rows = calculate()
    text = REPORT.read_text(encoding="utf-8")
    assert result_table(rows) in text and all(line in text for line in fingerprints()), "비용 비교 보고서가 현재 결과와 다름"
    return rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("계산 전 예상: " + PREDICTION, flush=True)
    paths = [p for p in ART.glob("*") if p.is_file()] + [ROOT / "scripts/train.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    valid, rows = calculate()
    table = result_table(rows)
    lines = ["# 비용 가정에 따른 임계값 비교 (6단계)", "",
             f"- 갱신일: {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d} (Asia/Seoul)",
             "- 변경 요약: 저장된 검증 예측으로 150·300·600만원 가정을 비교했다. 운영 기준은 바꾸지 않았다.",
             "- 재현 명령: `.\\.venv\\Scripts\\python.exe scripts/threshold_whatif.py`.",
             "- 근거: artifacts/validation_predictions.csv, artifacts/model_card.json, scripts/train.py의 pick_threshold.",
             "", "## 계산 전 예상", PREDICTION, "", "## 계산 방법",
             f"- 검증 기간: 2024년 7~11월, 정비 기록 {len(valid)}건, 실제 재고장 {int(valid.failure_within_30days.sum())}건.",
             "- 재점검 비용은 60만원으로 고정한다. 총비용 = 놓친 재고장 수 × 해당 비용 + 기준 이상 건수 × 60만원.",
             "- 기준 0.05~0.94를 0.01 간격으로 탐색하고 동률이면 낮은 기준을 고른다.",
             "", "## 결과", "", table, "", "## 검토",
             f"- 300만원 기준 {rows[1][1]:.2f}이 기존 모델 정보와 일치했다.",
             f"- 150만원 → 300만원 → 600만원 기준은 {rows[0][1]:.2f} → {rows[1][1]:.2f} → {rows[2][1]:.2f}이다. 비용이 커질수록 기준이 내려가는 예상과 일치한다.",
             "- 건수는 검증 기간 전체 정비 기록 기준이다. 드론 중복을 제거한 월 20대 선정이나 용량 제약을 반영한 최적화가 아니다.",
             "- 비용 가정별 총비용은 다른 단가로 계산하므로, 그 크기만으로 어느 가정이 더 좋은지 판단하지 않는다.",
             "- 동일 검증 기간에서 기준을 선택한 결과이며 독립 시험 성능이나 실제 비용 절감을 보장하지 않는다.",
             "- 현재 모델의 임계값과 모델 파일·학습 코드는 유지했다. 운영 비용 판단과 최종 결정은 정비 담당자가 한다.",
             "", "## 계산 근거 지문", ""] + fingerprints()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table)
    print(f"저장: {REPORT}\n모델·운영 기준·학습 코드 변경 없음")


if __name__ == "__main__":
    main()
