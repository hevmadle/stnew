"""쿠폰 캠페인 분석 대시보드.

1단계 — 캠페인 선택과 분석 가능성 확인: 캠페인 유형, 시작~종료일, 처치/대조 가구 수,
        분석 가능 상태(pipeline/rules.py 의 build_groups + 표본 부족 게이트).
2단계 — 분석 실행과 결과 조회: 캠페인과 발행 전 기간(pre_days)을 골라 실제 파이프라인
        (pipeline/prepare_data.py + estimate_effect.py)을 실행한다. 효과 결과보다
        성향점수 분포·공통지지영역·매칭률·매칭 전후 SMD를 먼저 보여주고(CLAUDE.md:
        품질 진단이 효과 결과보다 먼저 표시되어야 한다), 균형 게이트를 통과했을 때만
        주요·보조 효과와 95% 신뢰구간을 표시한다.

캐시: campaign_id·pre_days 가 이전 실행과 같으면(outputs/campaign_{id}/run_config.json
대조) 저장된 결과를 재사용하고 파이프라인을 다시 돌리지 않는다
(CLAUDE.md 파일 관리 규칙 5: 분석 재실행은 조건이 바뀔 때만 수행한다).

로직은 이 파일에서 새로 만들지 않고 pipeline/rules.py, prepare_data.py, estimate_effect.py
를 그대로 재사용한다.

실행:
    .venv/bin/streamlit run app/streamlit_app.py
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

import rules  # noqa: E402  (pipeline/rules.py — 집단 구성·매칭·게이트 규칙의 단일 출처)
from prepare_data import prepare  # noqa: E402
from estimate_effect import estimate  # noqa: E402
from profitability import Candidate, DataEstimate, compare_candidates  # noqa: E402

st.set_page_config(page_title="쿠폰 캠페인 분석", page_icon="🎯", layout="wide")

COLOR_TREATED = "#2a78d6"
COLOR_CONTROL = "#eb6834"
plt.rcParams.update({"font.family": "AppleGothic", "axes.unicode_minus": False})


# ---------------------------------------------------------------------
# 데이터 로드 (세션 내 1회만, 캠페인 30개 반복 조회에도 원본을 다시 읽지 않음)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="원본 데이터 불러오는 중...")
def load_shared_data():
    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    campaign_table = pd.read_csv(RAW_DIR / "campaign_table.csv")
    all_households = set(
        pd.read_csv(RAW_DIR / "transaction_data.csv", usecols=["household_key"])["household_key"].unique()
    )
    return campaign_desc, campaign_table, all_households


@st.cache_data(show_spinner=False)
def compute_feasibility(campaign_id: int, _campaign_desc, _campaign_table, _all_households) -> dict:
    """pipeline/rules.py의 build_groups()를 그대로 호출한다 — 규칙을 다시 만들지 않는다."""
    assignment, meta = rules.build_groups(campaign_id, _campaign_desc, _campaign_table, _all_households)
    n_treated = len(meta["treated"])
    n_control = len(meta["control"])
    n_excluded = len(meta["excluded"])

    ok = n_treated >= rules.MIN_GROUP_SIZE and n_control >= rules.MIN_GROUP_SIZE
    reasons = []
    if n_treated < rules.MIN_GROUP_SIZE:
        reasons.append(f"처치 {n_treated}명 < {rules.MIN_GROUP_SIZE}")
    if n_control < rules.MIN_GROUP_SIZE:
        reasons.append(f"대조 {n_control}명 < {rules.MIN_GROUP_SIZE}")

    return {
        "campaign_id": campaign_id,
        "description": meta["description"],
        "start_day": meta["start_day"],
        "end_day": meta["end_day"],
        "duration": meta["duration"],
        "overlapping_ids": meta["overlapping_ids"],
        "n_treated": n_treated,
        "n_control": n_control,
        "n_excluded": n_excluded,
        "control_treated_ratio": round(n_control / n_treated, 2) if n_treated > 0 else None,
        "feasible": ok,
        "reasons": reasons,
    }


@st.cache_data(show_spinner="30개 캠페인 전체 진단 계산 중...")
def compute_overview(_campaign_desc, _campaign_table, _all_households) -> pd.DataFrame:
    rows = []
    for cid in sorted(_campaign_desc["CAMPAIGN"].unique()):
        f = compute_feasibility(int(cid), _campaign_desc, _campaign_table, _all_households)
        rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 화면 구성
# ---------------------------------------------------------------------
st.title("🎯 쿠폰 캠페인 분석 — 캠페인 선택")
st.caption(
    "원본 데이터(`campaign_desc.csv`, `campaign_table.csv`, `transaction_data.csv`)에서 직접 조회·계산한 값입니다. "
    "이 화면은 표본 크기 기준의 1차 분석 가능성만 확인하며, 효과·수익성 결과는 다음 단계에서 다룹니다."
)

campaign_desc, campaign_table, all_households = load_shared_data()

with st.sidebar:
    st.header("캠페인 선택")
    campaign_ids = sorted(campaign_desc["CAMPAIGN"].unique())

    def format_campaign(cid: int) -> str:
        row = campaign_desc.loc[campaign_desc["CAMPAIGN"] == cid].iloc[0]
        return f"{cid}번 · {row['DESCRIPTION']} · DAY {row['START_DAY']}~{row['END_DAY']}"

    selected_id = st.selectbox("캠페인 번호", campaign_ids, format_func=format_campaign)
    st.caption(f"전체 캠페인 {len(campaign_ids)}개 · 전체 거래 가구 {len(all_households):,}명")

    feas = compute_feasibility(int(selected_id), campaign_desc, campaign_table, all_households)

    st.divider()
    st.header("분석 실행")
    pre_days = st.number_input(
        "발행 전 관찰 기간 (pre_days, 일)", min_value=1, max_value=700, value=90, step=10,
        help="캠페인 시작일 이전 며칠을 발행 전 특성 계산에 쓸지 정합니다.",
    )
    cached_now = rules.is_cached(int(selected_id), int(pre_days))
    if cached_now:
        st.caption("✅ 같은 조건의 저장된 결과가 있습니다 — 실행하면 캐시를 재사용합니다.")
    run_clicked = st.button(
        "분석 실행", type="primary", use_container_width=True, disabled=not feas["feasible"],
    )
    if not feas["feasible"]:
        st.caption("⚠️ 표본 부족으로 실행할 수 없습니다.")

st.subheader(f"캠페인 {feas['campaign_id']}번 상세 — {feas['description']}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("캠페인 유형", feas["description"])
col2.metric("시작일 (DAY)", feas["start_day"])
col3.metric("종료일 (DAY)", feas["end_day"])
col4.metric("기간", f"{feas['duration']}일")

col5, col6, col7, col8 = st.columns(4)
col5.metric("처치 가구 수", f"{feas['n_treated']:,}")
col6.metric("대조군 후보 수", f"{feas['n_control']:,}")
col7.metric(
    "대조/처치 비율",
    f"{feas['control_treated_ratio']:.2f}" if feas["control_treated_ratio"] is not None else "—",
)
with col8:
    if feas["feasible"]:
        st.metric("분석 가능 상태", "가능")
        st.success("표본 크기 기준 분석 가능")
    else:
        st.metric("분석 가능 상태", "어려움")
        st.error("표본 부족: " + ", ".join(feas["reasons"]))

with st.expander("겹치는 캠페인·제외 가구 자세히 보기"):
    st.write(
        f"기간이 겹치는 다른 캠페인 **{len(feas['overlapping_ids'])}개**: "
        f"{feas['overlapping_ids'] if feas['overlapping_ids'] else '없음'}"
    )
    st.write(
        f"제외 가구 **{feas['n_excluded']:,}명** — 처치·대조 정의(같은 기간 겹치는 다른 캠페인 미노출)에 "
        f"맞지 않아 빠진 가구입니다. 캠페인을 한 번도 받은 적 없는 가구도 대조군에서 제외했습니다."
    )
    st.caption(
        "처치 = 이 캠페인을 받고 겹치는 다른 캠페인은 받지 않은 가구 · "
        "대조 = 이 캠페인도 겹치는 캠페인도 받지 않았지만 다른 캠페인 이력은 있는 가구 "
        "(pipeline/rules.py의 build_groups 규칙, CLAUDE.md 분석 설계 규칙 1·2)"
    )

st.divider()

# =======================================================================
# 분석 실행 및 결과 조회
# =======================================================================
st.subheader("📊 분석 실행 및 결과")

OUTCOME_LABELS = {
    "target_purchase": "대상 상품 구매율", "target_sales": "대상 상품 구매금액", "target_quantity": "대상 상품 구매수량",
    "any_purchase": "전체 구매율", "total_sales": "전체 구매금액", "baskets": "장바구니 수",
}
session_key = f"analysis_{int(selected_id)}_{int(pre_days)}"

if run_clicked:
    campaign_id_run = int(selected_id)
    pre_days_run = int(pre_days)
    from_cache = rules.is_cached(campaign_id_run, pre_days_run)
    log_buffer = io.StringIO()

    with st.spinner("저장된 결과를 불러오는 중..." if from_cache else "파이프라인 실행 중 — 성향점수·매칭·효과 추정 (몇 초 걸릴 수 있습니다)"):
        if from_cache:
            with open(rules.output_dir(campaign_id_run) / "results.json", encoding="utf-8") as f:
                results = json.load(f)
        else:
            with redirect_stdout(log_buffer):
                prep_df, prep_meta, prep_gate = prepare(campaign_id_run, pre_days_run)
                if prep_df is None:
                    results = {
                        "status": "중단", "stage": "prepare_data",
                        "reason": prep_gate["reason"], "campaign_id": campaign_id_run,
                    }
                else:
                    out_dir = rules.output_dir(campaign_id_run)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    prep_df.to_csv(out_dir / "analysis_data.csv", index=False)
                    rules.save_run_config(campaign_id_run, pre_days_run)
                    results = estimate(campaign_id_run)
                    if results.get("status") == "중단":
                        results["stage"] = "estimate_effect"

    st.session_state[session_key] = {"results": results, "from_cache": from_cache, "log": log_buffer.getvalue()}

state = st.session_state.get(session_key)

if state is None:
    st.info("왼쪽 사이드바에서 발행 전 관찰 기간(pre_days)을 정하고 **분석 실행**을 눌러주세요.")
else:
    results = state["results"]
    campaign_id_shown = int(selected_id)

    st.caption("✅ 저장된 결과를 재사용했습니다 (같은 조건 캐시)" if state["from_cache"] else "🔄 새로 계산했습니다")
    if state["log"]:
        with st.expander("실행 로그 보기"):
            st.code(state["log"], language=None)

    if results.get("status") == "중단":
        st.error(f"❌ 파이프라인 중단 ({results.get('stage', '?')} 단계) — {results.get('reason')}")
        partial_sample = results.get("sample")
        if partial_sample and "common_support" in partial_sample:
            cs = partial_sample["common_support"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("공통지지영역 하한", f"{cs['lo']:.4f}")
            c2.metric("공통지지영역 상한", f"{cs['hi']:.4f}")
            c3.metric("영역 내 처치", cs["n_treated_in_support"])
            c4.metric("영역 내 대조", cs["n_control_in_support"])
            st.caption("게이트를 통과하지 못해 이 시점까지만 계산됐습니다.")
    else:
        analysis_path = rules.output_dir(campaign_id_shown) / "analysis_data.csv"
        analysis_df = pd.read_csv(analysis_path)
        sample = results["sample"]
        cs = sample["common_support"]
        m = sample["matching"]
        balance = results["balance"]

        # --- 1. 성향점수 분포 ---------------------------------------------------
        st.markdown("#### 1️⃣ 성향점수 분포")
        t_scores = analysis_df.loc[analysis_df["treatment"] == 1, "p_score"]
        c_scores = analysis_df.loc[analysis_df["treatment"] == 0, "p_score"]

        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
        bins = np.linspace(0, 1, 41)
        ax.axvspan(0, cs["lo"], color="#898781", alpha=0.12, zorder=0)
        ax.axvspan(cs["hi"], 1, color="#898781", alpha=0.12, zorder=0)
        ax.hist(t_scores, bins=bins, color=COLOR_TREATED, alpha=0.55, density=True,
                label=f"처치 (n={len(t_scores)})", zorder=3)
        ax.hist(c_scores, bins=bins, color=COLOR_CONTROL, alpha=0.55, density=True,
                label=f"대조 (n={len(c_scores)})", zorder=2)
        ax.axvline(cs["lo"], color="#52514e", linewidth=1.2, linestyle="--", zorder=4)
        ax.axvline(cs["hi"], color="#52514e", linewidth=1.2, linestyle="--", zorder=4)
        ax.set_xlabel("성향점수 (p_score)")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("회색 음영 = 공통지지영역 밖 (점선 = 영역 경계)")

        # --- 2. 공통지지영역 -----------------------------------------------------
        st.markdown("#### 2️⃣ 공통지지영역")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("하한", f"{cs['lo']:.4f}")
        c2.metric("상한", f"{cs['hi']:.4f}")
        c3.metric("영역 내 처치", f"{cs['n_treated_in_support']:,} / {sample['treated_total']:,}")
        c4.metric("영역 내 대조", f"{cs['n_control_in_support']:,} / {sample['control_total']:,}")

        # --- 3. 매칭률 -----------------------------------------------------------
        st.markdown("#### 3️⃣ 매칭률")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("선택된 caliper", f"{m['caliper_multiplier']}×SD")
        c2.metric("매칭쌍 수", f"{m['n_pairs']:,}")
        c3.metric("처치 매칭률", f"{m['treated_match_rate']:.1%}")
        c4.metric("발행 전 관찰기간", f"{results.get('pre_days', pre_days)}일")

        # --- 4. 매칭 전후 SMD -----------------------------------------------------
        st.markdown("#### 4️⃣ 매칭 전후 SMD (표준화평균차)")
        smd_df = pd.DataFrame({
            "매칭전": pd.Series(balance["smd_before"]).abs(),
            "매칭후": pd.Series(balance["smd_after"]).abs(),
        })
        st.bar_chart(smd_df)
        st.dataframe(smd_df.round(4), use_container_width=True)
        st.caption(
            f"매칭 후 최대|SMD| = {balance['max_abs_smd_after']:.3f} "
            f"(기준 {balance['threshold']}, 매칭 전 최대 {balance['max_abs_smd_before']:.3f})"
        )

        # --- 품질 상태 배너 --------------------------------------------------------
        st.divider()
        if results["status"] == "완료":
            st.success(f"✅ 품질 게이트 전부 통과 — 상태: {results['status']}")
        else:
            st.warning(f"⚠️ {results['status']} — {results.get('reason', '')}")

        # --- 5. 효과 결과 (품질 게이트를 통과했을 때만) -------------------------------
        st.markdown("#### 5️⃣ 효과 결과 (매칭 전 단순차이 vs 매칭 후, 95% CI)")
        if results["status"] != "완료":
            st.info(
                "품질 게이트를 완전히 통과하지 못해 효과 결과를 표시하지 않습니다 "
                "(CLAUDE.md: 큰 잔여 불균형이 있으면 효과를 확정하지 않는다). "
                f"사유: {results.get('reason', '균형 기준 미달')}"
            )
        else:
            effects = results["effects"]
            rows = []
            for bucket, tag in [("primary_target_product", "주요(대상상품)"), ("secondary_overall", "보조(전체)")]:
                for col, e in effects[bucket].items():
                    rows.append({
                        "구분": tag, "변수": OUTCOME_LABELS.get(col, col),
                        "매칭전 차이": e["unadjusted_diff"],
                        "매칭전 95% CI": f"[{e['unadjusted_ci95'][0]:.3f}, {e['unadjusted_ci95'][1]:.3f}]",
                        "매칭후 차이": e["matched_diff"],
                        "매칭후 95% CI": f"[{e['matched_ci95'][0]:.3f}, {e['matched_ci95'][1]:.3f}]",
                    })
            effect_df = pd.DataFrame(rows)
            st.dataframe(effect_df, use_container_width=True, hide_index=True)
            st.caption(
                "매칭전 = 전체 표본 독립비교(Welch CI) · 매칭후 = 매칭쌍 대응비교(paired CI). "
                "단순 차이이며 인과효과로 확정하지 않습니다(CLAUDE.md 품질과 해석 규칙 2)."
            )

        # --- 6. 쿠폰 후보안 비교 (같은 캠페인 효과를 공유) -------------------------------
        st.markdown("#### 6️⃣ 쿠폰 후보안 비교")

        target_sales_effect = results["effects"]["primary_target_product"]["target_sales"]
        data_estimate = DataEstimate(
            campaign_id=campaign_id_shown,
            status=results["status"],
            outcome="target_sales",
            incremental_revenue_per_customer=target_sales_effect["matched_diff"],
            incremental_revenue_ci95=tuple(target_sales_effect["matched_ci95"]),
            n_matched_pairs=sample["matching"]["n_pairs"],
            n_treated_total=sample["treated_total"],
        )

        if results["status"] != "완료":
            st.warning(
                f"⚠️ 효과 추정 상태가 '{results['status']}'입니다 — 아래 수익성 계산은 참고용이며 "
                "확정된 값으로 쓰지 마세요."
            )
        st.caption(
            f"모든 후보에 공통 적용: 고객당 증분매출(매칭 후, target_sales) "
            f"{data_estimate.incremental_revenue_per_customer:,.2f}  "
            f"95% CI [{data_estimate.incremental_revenue_ci95[0]:,.2f}, "
            f"{data_estimate.incremental_revenue_ci95[1]:,.2f}]"
        )

        gross_margin_rate = st.slider(
            "매출총이익률 (모든 후보 공통 가정)", 0.0, 1.0, 0.30, 0.01,
            key=f"margin_{campaign_id_shown}",
        )

        default_n = data_estimate.n_treated_total or 500
        default_candidates = pd.DataFrame([
            {"후보": "후보 1", "할인액": 2.0, "예상사용률": 0.30, "발행수": default_n, "운영비": 50000.0},
            {"후보": "후보 2", "할인액": 3.0, "예상사용률": 0.40, "발행수": default_n, "운영비": 50000.0},
        ])
        edited = st.data_editor(
            default_candidates,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"candidates_{campaign_id_shown}",
            column_config={
                "할인액": st.column_config.NumberColumn(min_value=0.0, step=0.5, help="쿠폰당 할인비용"),
                "예상사용률": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05),
                "발행수": st.column_config.NumberColumn(min_value=1, step=10),
                "운영비": st.column_config.NumberColumn(min_value=0.0, step=1000.0),
            },
        )

        valid_rows = edited.dropna(subset=["후보", "할인액", "예상사용률", "발행수", "운영비"])
        candidates = [
            Candidate(
                label=str(row["후보"]),
                coupon_cost_per_redemption=float(row["할인액"]),
                redemption_rate=float(row["예상사용률"]),
                n_issued=int(row["발행수"]),
                operating_cost=float(row["운영비"]),
            )
            for _, row in valid_rows.iterrows()
        ]

        if not candidates:
            st.info("표에 후보를 1개 이상 입력하세요.")
        else:
            try:
                comp_df, recommended, comp_results = compare_candidates(
                    data_estimate, gross_margin_rate, candidates
                )
            except ValueError as e:
                st.error(f"입력값을 확인하세요: {e}")
            else:
                display_cols = [
                    "후보", "할인액", "예상사용률", "발행수", "운영비",
                    "예상증분매출", "총비용", "증분이익", "증분이익95%CI", "ROI",
                    "손익분기충족", "주의건수", "추천",
                ]
                st.dataframe(
                    comp_df[display_cols].style.apply(
                        lambda r: ["background-color: #eaf7ea" if r["추천"] else "" for _ in r], axis=1
                    ),
                    use_container_width=True, hide_index=True,
                )
                if recommended:
                    st.success(
                        f"⭐ 추천: **{recommended}** — 입력한 후보군 중 손익분기(증분이익 ≥ 0)를 만족하면서 "
                        "증분이익이 가장 큰 안입니다. 이 후보군 밖의 다른 조합까지 포함한 전역 최적은 아닙니다."
                    )
                else:
                    st.error("입력한 후보군 중 손익분기(증분이익 ≥ 0)를 만족하는 안이 없습니다.")

                if any(r.warnings for r in comp_results):
                    with st.expander("후보별 주의사항 보기"):
                        for c, r in zip(candidates, comp_results):
                            if r.warnings:
                                st.markdown(f"**{c.label}**")
                                for w in r.warnings:
                                    st.caption(f"⚠ {w}")

st.divider()

st.subheader("전체 캠페인 한눈에 보기")
overview = compute_overview(campaign_desc, campaign_table, all_households)
display_df = overview.copy()
display_df["기간"] = display_df["start_day"].astype(str) + "~" + display_df["end_day"].astype(str)
display_df["상태"] = display_df["feasible"].map({True: "✅ 가능", False: "⚠️ 어려움"})
display_df = display_df.rename(columns={
    "campaign_id": "캠페인", "description": "유형", "duration": "기간(일)",
    "n_treated": "처치", "n_control": "대조", "n_excluded": "제외",
    "control_treated_ratio": "대조/처치비율",
})
display_df = display_df[["캠페인", "유형", "기간", "기간(일)", "처치", "대조", "제외", "대조/처치비율", "상태"]]

st.dataframe(
    display_df.style.apply(
        lambda r: ["background-color: #fdecea" if r["상태"] == "⚠️ 어려움" else "" for _ in r], axis=1
    ),
    use_container_width=True,
    hide_index=True,
)

n_feasible = int(overview["feasible"].sum())
st.caption(
    f"분석 가능(표본 기준) {n_feasible}개 / 전체 {len(overview)}개 — "
    f"기준: 처치·대조 각각 {rules.MIN_GROUP_SIZE}가구 이상 (pipeline/rules.py MIN_GROUP_SIZE)"
)
