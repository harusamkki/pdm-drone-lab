"""실행: streamlit run app.py. 저장된 드론 예측 결과를 조회한다."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict import validate_outputs
from word_export import markdown_to_docx
from upload_data import REQUIRED, score_upload
from dashboard_analysis import context, concentration, capacity_comparison, monthly_history, month_changes, quality

st.set_page_config(page_title="드론 정비 우선순위", page_icon="🛠️", layout="wide")
st.caption("정비 운영 대시보드 · 교육용 가상 데이터")
st.title("드론 정비 우선순위")
st.caption("드론별 최신 정비 기록 기준 · 각 정비 후 30일 재고장 예측 · 실습 대상 월")

try:
    score = validate_outputs()
    card = json.loads((ROOT / "artifacts/model_card.json").read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"저장 결과 검증에 실패했습니다: {exc}")
    st.info("터미널에서 python scripts/check.py로 상태를 확인하세요.")
    st.stop()

with st.sidebar:
    st.header("분석 데이터")
    upload = st.file_uploader("월별 정비 기록 CSV 업로드", type=["csv"], help="기존 월별 드론 정비 CSV와 같은 컬럼을 사용하세요. UTF-8·CP949 지원.")
    st.download_button("입력 양식 CSV", pd.DataFrame(columns=REQUIRED).to_csv(index=False).encode("utf-8-sig"),
                       "월별_정비_입력양식.csv", "text/csv")
    st.caption("업로드를 지우면 기본 2026년 1월 데이터로 돌아갑니다. 파일은 서버에 저장하지 않습니다.")
if upload is not None:
    try:
        score = score_upload(upload.getvalue(), card)
    except Exception as exc:
        st.error(f"업로드 파일을 분석할 수 없습니다: {exc}")
        st.info("필수 컬럼·날짜·숫자 형식을 확인하거나 입력 양식 CSV를 이용하세요.")
        st.stop()
    st.info("업로드 데이터의 예측 결과입니다. 기존 학습 모델을 사용하며 자동 재학습하지 않습니다.")
    for message in score["warnings"]:
        st.warning(message)
st.subheader(f"{score['period']} 재점검 계획")
with st.sidebar:
    st.header("점검 계획 설정")
    if score["devices"] == 1:
        cap = 1
        st.caption("재점검 가능 대수: 1대")
    else:
        cap = st.slider("재점검 가능 대수", 1, score["devices"], score["capacity"])
    threshold = st.slider("위험 표시 기준", .05, .95, float(score["threshold"]), .01)
    st.caption("설정 변경은 화면의 비교용이며 저장된 명단·보고서는 바꾸지 않습니다.")
    st.caption("데이터: " + (upload.name if upload is not None else "data/drone_this_month.csv"))

ranked = pd.DataFrame(score["rows"])
ranked["재점검"] = ranked["rank"].le(cap)
ranked["위험 높음"] = ranked["prob"].ge(threshold)
c1, c2, c3 = st.columns(3)
c1.metric("정비 드론", f"{len(ranked)}대", border=True)
c2.metric("재점검 대상", f"{int(ranked['재점검'].sum())}대", border=True)
c3.metric("위험 높음", f"{int(ranked['위험 높음'].sum())}대", border=True)
st.caption(f"정비 기록 {score['records']}건 · 위험 기준 {threshold:.0%} · 확률 상위 {cap}대 선정")

tab1, tab2, tab3, tab5, tab6, tab7, tab8, tab9, tab4 = st.tabs([
    "재점검 목록", "드론별 근거", "모델 검증",
    "정비 이력", "위험 집중 분석", "점검 용량 비교", "월별 변화", "데이터 품질", "보고서·제출 파일"])
with tab1:
    view = ranked[["rank", "재점검", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "prob", "위험 높음", "reasons"]].rename(columns={
        "rank": "순위", "equipment_id": "드론", "equipment_type": "유형", "unit_code": "부대",
        "maintenance_date": "최근 정비일", "prob": "재고장 확률(%)", "reasons": "위험을 올린 요인"})
    view["재고장 확률(%)"] *= 100
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "재고장 확률(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
    st.download_button("현재 설정의 재점검 목록 CSV", view[view["재점검"]].to_csv(index=False).encode("utf-8-sig"),
                       f"{score['period']}_재점검_목록.csv", "text/csv")
    if upload is not None:
        st.download_button("업로드 데이터 전체 순위 CSV", ranked.to_csv(index=False).encode("utf-8-sig"),
                           f"{score['period']}_전체_예측.csv", "text/csv")
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
    if upload is not None:
        st.info("아래 문서는 기존 교육용 2026년 1월 자료입니다. 업로드 결과는 재점검 목록 탭의 CSV로 내려받으세요.")
    for label, name, mime in [
        ("2025년 시험 예측 CSV", "test_predictions_2025.csv", "text/csv"),
        ("분석 보고서", "analysis_report.md", "text/markdown"),
        ("재점검 계획 보고서", "2026-01_재점검계획.md", "text/markdown"),
        ("대시보드 추가 분석", "dashboard_analysis.md", "text/markdown"),
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

raw, history = context(score)
with tab5:
    st.subheader("드론별 정비 이력 타임라인")
    selected_drone = st.selectbox("이력을 볼 드론", ranked.equipment_id, key="history_drone")
    events = history.loc[history.equipment_id.eq(selected_drone)].sort_values("maintenance_date").copy()
    chart = alt.Chart(events).mark_circle(size=100).encode(
        x=alt.X("maintenance_date:T", title="정비일"),
        y=alt.Y("severity:Q", title="고장 심각도"),
        color=alt.Color("maintenance_type:N", title="정비 구분"),
        tooltip=["maintenance_date:T", "maintenance_type:N", "fault_category:N",
                 "maintenance_success:Q", "fail_count_90d:Q", "hours_since_last_pm:Q"])
    st.altair_chart(chart, width="stretch")
    st.dataframe(events[["maintenance_date", "maintenance_type", "fault_category", "maintenance_success", "fail_count_90d", "hours_since_last_pm"]].rename(
        columns={"maintenance_date": "정비일", "maintenance_type": "정비 구분", "fault_category": "고장 유형",
                 "maintenance_success": "정비 성공(1/0)", "fail_count_90d": "최근 90일 고장 수", "hours_since_last_pm": "예방정비 후 비행시간"}),
        hide_index=True, width="stretch")
    st.caption("기존 제공 이력과 업로드 기록을 연결한 관측 이력입니다. 대상 월 이후의 기록은 표시하지 않습니다.")
with tab6:
    st.subheader("위험 드론 집중 분석")
    grouping = st.radio("집계 기준", ["드론 유형", "부대"], horizontal=True)
    field = "equipment_type" if grouping == "드론 유형" else "unit_code"
    grouped = concentration(ranked, field).rename(columns={field: grouping})
    st.dataframe(grouped, hide_index=True, width="stretch")
    st.bar_chart(grouped, x=grouping, y="위험 비율(%)")
    st.caption("위험 비율 = 현재 기준 이상 드론 수 ÷ 해당 집단의 정비 드론 수. 실제 고장률이 아닙니다. 소수 드론 집단은 비율 변동이 클 수 있습니다.")
with tab7:
    st.subheader("점검 용량 시뮬레이션")
    capacity_table = capacity_comparison(ranked, threshold, cap)
    st.dataframe(capacity_table, hide_index=True, width="stretch")
    st.line_chart(capacity_table.set_index("점검 가능 대수")[["위험 드론 포함", "미선정 위험 드론"]])
    st.caption("같은 위험 순위에서 점검 대수만 바꾼 비교입니다. 예방한 고장 수·실제 적중 수·절감액을 의미하지 않습니다. 위험 드론이 없으면 포함률은 계산하지 않습니다.")
with tab8:
    st.subheader("월별 위험 변화")
    monthly = monthly_history(history, score["period"])
    changes = month_changes(ranked, monthly, score["period"])
    displayed_changes = changes.rename(columns={"equipment_id": "드론", "prob": "현재 예측", "previous_prob": "전월 예측",
                                               "previous_fail_count_90d": "전월 최근 90일 고장 수", "fail_count_90d": "현재 최근 90일 고장 수"})
    st.dataframe(displayed_changes, hide_index=True, width="stretch", column_config={
        "현재 예측": st.column_config.NumberColumn(format="percent"),
        "전월 예측": st.column_config.NumberColumn(format="percent")})
    trend_drone = st.selectbox("변화를 볼 드론", ranked.equipment_id, key="trend_drone")
    trend = monthly.loc[monthly.equipment_id.eq(trend_drone), ["month", "prob", "fail_count_90d"]].copy()
    trend["예측값(%)"] = trend.prob * 100
    st.line_chart(trend.set_index("month")[["예측값(%)"]])
    st.dataframe(trend.rename(columns={"month": "월", "fail_count_90d": "최근 90일 고장 수"}).drop(columns="prob"), hide_index=True)
    st.caption("최근 6개월 중 정비 기록이 있는 달만 비교합니다. 전월 기록이 없으면 증감은 미확인입니다. 고정된 최종 모델을 과거 월에도 적용한 조회용 값이며, 당시 예측을 보관한 결과나 독립적인 과거 성능 검증이 아닙니다.")
with tab9:
    st.subheader("입력 데이터 품질 점검")
    checks = quality(raw, history)
    qa, qb, qc = st.columns(3)
    qa.metric("전체 결측 셀", int(raw.isna().sum().sum()))
    qb.metric("드론·정비일 중복", checks["duplicates"])
    qc.metric("이전 이력 없는 드론", len(checks["no_history"]))
    st.dataframe(checks["missing"], hide_index=True, width="stretch")
    st.write(f"처리 가정 대상: 운용중단 999 {checks['sentinel']}건 / 지정 비용 극단값 {checks['cost_candidates']}건")
    st.write("학습 데이터 범위와 비교")
    st.dataframe(checks["ranges"], hide_index=True, width="stretch")
    st.dataframe(checks["unknown"], hide_index=True, width="stretch")
    if checks["no_history"]:
        st.write("이전 이력 없는 드론: " + ", ".join(checks["no_history"]))
    st.caption("원본 입력 기준 점검입니다. 예방정비의 고장 정보 결측은 정상 문맥일 수 있습니다. 학습 범위 밖 값이나 새 범주가 반드시 오류인 것은 아니며, 장비 노후화·비행시간 증가도 원인이 될 수 있습니다.")

st.divider()
st.subheader("부대별 요약")
summary = ranked.groupby("unit_code").agg(**{
    "정비 드론 수": ("equipment_id", "size"), "재점검 대상 수": ("재점검", "sum"),
    "위험 높음 수": ("위험 높음", "sum"),
}).reset_index().rename(columns={"unit_code": "부대"}).sort_values(
    ["재점검 대상 수", "부대"], ascending=[False, True]).reset_index(drop=True)
st.dataframe(summary, hide_index=True, width="stretch")
st.write(f"재점검 대상 합계: {int(summary['재점검 대상 수'].sum())}대")
