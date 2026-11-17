"""실행: streamlit run app.py. 저장된 드론 예측 결과를 조회한다."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict import validate_outputs
from word_export import markdown_to_docx

st.set_page_config(page_title="드론 정비 우선순위", page_icon="🛠️", layout="wide")
st.caption("정비 운영 대시보드 · 교육용 가상 데이터")
st.title("드론 정비 우선순위")
st.subheader("2026년 1월 재점검 계획")
st.caption("드론별 최신 정비 기록 기준 · 각 정비 후 30일 재고장 예측 · 실습 대상 월")

try:
    score = validate_outputs()
    card = json.loads((ROOT / "artifacts/model_card.json").read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"저장 결과 검증에 실패했습니다: {exc}")
    st.info("터미널에서 python scripts/check.py로 상태를 확인하세요.")
    st.stop()

with st.sidebar:
    st.header("점검 계획 설정")
    cap = st.slider("재점검 가능 대수", 1, score["devices"], score["capacity"])
    threshold = st.slider("위험 표시 기준", .05, .95, float(score["threshold"]), .01)
    st.caption("설정 변경은 화면의 비교용이며 저장된 명단·보고서는 바꾸지 않습니다.")
    st.caption("데이터: data/drone_this_month.csv")

ranked = pd.DataFrame(score["rows"])
ranked["재점검"] = ranked["rank"].le(cap)
ranked["위험 높음"] = ranked["prob"].ge(threshold)
c1, c2, c3 = st.columns(3)
c1.metric("정비 드론", f"{len(ranked)}대", border=True)
c2.metric("재점검 대상", f"{int(ranked['재점검'].sum())}대", border=True)
c3.metric("위험 높음", f"{int(ranked['위험 높음'].sum())}대", border=True)
st.caption(f"정비 기록 {score['records']}건 · 위험 기준 {threshold:.0%} · 확률 상위 {cap}대 선정")

tab1, tab2, tab3, tab4 = st.tabs(["재점검 목록", "드론별 근거", "모델 검증", "보고서·제출 파일"])
with tab1:
    view = ranked[["rank", "재점검", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "prob", "위험 높음", "reasons"]].rename(columns={
        "rank": "순위", "equipment_id": "드론", "equipment_type": "유형", "unit_code": "부대",
        "maintenance_date": "최근 정비일", "prob": "재고장 확률(%)", "reasons": "위험을 올린 요인"})
    view["재고장 확률(%)"] *= 100
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "재고장 확률(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
    st.download_button("현재 설정의 재점검 목록 CSV", view[view["재점검"]].to_csv(index=False).encode("utf-8-sig"),
                       "재점검_목록.csv", "text/csv")
with tab2:
    drone = st.selectbox("드론 선택", ranked.equipment_id)
    row = ranked.loc[ranked.equipment_id.eq(drone)].iloc[0]
    st.metric("재고장 예측 확률", f"{row.prob:.1%}")
    st.write(f"최근 정비일: {row.maintenance_date} · 전체 순위: {row['rank']}")
    st.write("위험을 올린 주요 요인: " + row.reasons)
    st.caption("모델의 양의 SHAP 기여 상위 요인이며 고장의 원인이나 점검 효과를 입증하지 않습니다.")
with tab3:
    metrics = card["validation"]
    st.write("평가 기간: 2024년 7~11월. 2025년 시험 성능은 정답이 없어 미확인입니다.")
    a, b, c = st.columns(3)
    a.metric("검증 PR-AUC (AP)", f"{metrics['pr_auc_ap']:.4f}")
    b.metric("검증 재고장 비율", f"{metrics['base_rate']:.4f}")
    c.metric("검증 ROC-AUC", f"{metrics['roc_auc']:.4f}")
    st.dataframe(pd.read_csv(ROOT / "artifacts/backtest_validation.csv"), hide_index=True, width="stretch")
    st.caption("월별 드론의 마지막 기록만 남겨 20대씩 선택한 적중 수입니다.")
    for limit in card["limits"]:
        st.write("- " + limit)
with tab4:
    for label, name, mime in [
        ("2025년 시험 예측 CSV", "test_predictions_2025.csv", "text/csv"),
        ("분석 보고서", "analysis_report.md", "text/markdown"),
        ("재점검 계획 보고서", "2026-01_재점검계획.md", "text/markdown"),
    ]:
        path = ROOT / "reports" / name
        if path.exists():
            if path.suffix == ".md":
                st.download_button(label + " (Word)", markdown_to_docx(path.read_text(encoding="utf-8")),
                                   file_name=path.with_suffix(".docx").name,
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            else:
                st.download_button(label, path.read_bytes(), file_name=name, mime=mime)
    st.caption("저장 파일은 기본 20대·기준 26% 결과입니다. 사이드바 변경은 저장 파일에 반영되지 않습니다.")

st.divider()
st.subheader("부대별 요약")
summary = ranked.groupby("unit_code").agg(**{
    "정비 드론 수": ("equipment_id", "size"), "재점검 대상 수": ("재점검", "sum"),
    "위험 높음 수": ("위험 높음", "sum"),
}).reset_index().rename(columns={"unit_code": "부대"}).sort_values(
    ["재점검 대상 수", "부대"], ascending=[False, True]).reset_index(drop=True)
st.dataframe(summary, hide_index=True, width="stretch")
st.write(f"재점검 대상 합계: {int(summary['재점검 대상 수'].sum())}대")
