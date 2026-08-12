"""캠페인 18번 발행 전 특성 — 처치 vs 대조 비교표 + 분포 차이 그래프.

검증된 outputs/campaign_18/analysis_data.csv 를 그대로 읽어(재계산하지 않음) 발행 전
특성 11개를 처치집단(510)·대조집단(335)으로 비교한다.

    - 연속 변수: 평균, 이진 변수: 비율(=평균과 동일 계산이나 %로 표기)
    - 평균차이 = 처치평균 - 대조평균
    - 표준화평균차(SMD) = 평균차이 / sqrt((분산_처치 + 분산_대조)/2)  — 매칭 전 불균형의 표준 지표

|SMD| >= 0.3(중간 이상 차이, Cohen 관례) 인 변수는 분포 그래프로 그린다.
이 단계는 성향점수·매칭과 무관한 기술 통계이며, 어떤 변수도 "효과"로 표현하지 않는다.

산출:
    outputs/campaign_18/pretreatment_balance_smd.png   : 변수별 SMD 순위 그래프 (Love plot)
    outputs/campaign_18/pretreatment_balance_dist.png  : |SMD|>=0.3 변수의 분포 비교

CLAUDE.md 파일 관리 규칙 1(불필요한 파일 지양)에 따라 이 두 PNG는 사용자가 명시적으로
요청한 그래프 산출물이며, 캠페인별 4대 기본 파일(analysis_data.csv 등)과는 별개의
진단용 이미지다.

실행:
    .venv/bin/python analysis/campaign_18_pretreatment_balance.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN
from campaign_18_pre_features import PRE_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}"
INPUT_FILE = OUTPUT_DIR / "analysis_data.csv"

SMD_CHART_THRESHOLD = 0.3  # 이 이상이면 분포 그래프에 포함
BINARY_COLUMNS = {"pre_target_purchase"}

LABELS = {
    "recency": "최근 구매 이후 경과일(recency)",
    "pre_baskets": "장바구니 수",
    "pre_sales": "구매금액",
    "pre_quantity": "구매수량",
    "pre_active_days": "구매 활동일수",
    "pre_target_purchase": "대상 상품 구매 비율",
    "pre_target_baskets": "대상 상품 장바구니 수",
    "pre_target_sales": "대상 상품 구매금액",
    "pre_target_quantity": "대상 상품 구매수량",
    "pre_coupon_redemptions": "쿠폰 사용 횟수",
    "pre_campaign_count": "캠페인 수신 횟수",
}

# 색상 — dataviz 스킬 기본 팔레트 카테고리컬 slot1(blue)/slot2(orange), CVD 안전 검증됨
COLOR_TREATED = "#2a78d6"
COLOR_CONTROL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "grid.color": GRIDLINE,
})


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "검증된 분석표 불러오기")
    # ------------------------------------------------------------------
    if not INPUT_FILE.exists():
        raise SystemExit(f"{INPUT_FILE} 가 없다. analysis/campaign_18_build_analysis_table.py 를 먼저 실행하라.")

    df = pd.read_csv(INPUT_FILE)
    t = df[df["treatment"] == 1]
    c = df[df["treatment"] == 0]
    print(f"  입력 파일: {INPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  가구 수  : 처치 {len(t):,} / 대조 {len(c):,}")

    # ------------------------------------------------------------------
    banner("STEP 2", "발행 전 특성 비교표 — 평균/비율, 평균차이, SMD")
    # ------------------------------------------------------------------
    rows = []
    for col in PRE_FEATURE_COLUMNS:
        mean_t, mean_c = t[col].mean(), c[col].mean()
        var_t, var_c = t[col].var(ddof=1), c[col].var(ddof=1)
        pooled_sd = np.sqrt((var_t + var_c) / 2)
        diff = mean_t - mean_c
        smd = diff / pooled_sd if pooled_sd > 0 else np.nan
        is_binary = col in BINARY_COLUMNS
        rows.append(
            {
                "변수": LABELS[col],
                "열": col,
                "유형": "비율(0/1)" if is_binary else "평균(연속)",
                "처치": round(mean_t * 100, 1) if is_binary else round(mean_t, 2),
                "대조": round(mean_c * 100, 1) if is_binary else round(mean_c, 2),
                "단위": "%" if is_binary else "",
                "평균차이": round(diff, 3) if not is_binary else round(diff * 100, 1),
                "SMD": round(smd, 3) if pd.notna(smd) else None,
            }
        )
    balance = pd.DataFrame(rows).sort_values("SMD", key=lambda s: s.abs(), ascending=False, na_position="last")

    pd.set_option("display.width", 200)
    print(balance.drop(columns=["열"]).to_string(index=False))

    n_constant = balance["SMD"].isna().sum()
    if n_constant:
        const_vars = balance.loc[balance["SMD"].isna(), "변수"].tolist()
        print(f"\n  ※ 분산이 0이라 SMD를 계산할 수 없는 변수: {const_vars} "
              f"(두 집단 모두 100% 동일값 — 대상 상품 구매 이력이 전원에게 있음)")

    # ------------------------------------------------------------------
    banner("STEP 3", f"분포 차이가 큰 변수 선별 (|SMD| >= {SMD_CHART_THRESHOLD})")
    # ------------------------------------------------------------------
    large = balance[balance["SMD"].abs() >= SMD_CHART_THRESHOLD].copy()
    print(f"  {len(large)}개 변수: {large['변수'].tolist()}")
    small = balance[(balance["SMD"].abs() < SMD_CHART_THRESHOLD) & balance["SMD"].notna()]
    print(f"  차이가 작은 변수({len(small)}개, |SMD|<{SMD_CHART_THRESHOLD}): {small['변수'].tolist()}")

    # ------------------------------------------------------------------
    banner("STEP 4", "그래프 1 — 변수별 SMD 순위 (Love plot)")
    # ------------------------------------------------------------------
    plot_df = balance.dropna(subset=["SMD"]).sort_values("SMD")
    colors = [COLOR_TREATED if v >= 0 else COLOR_CONTROL for v in plot_df["SMD"]]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["SMD"], color=colors, height=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["변수"], fontsize=9)
    ax.axvline(0, color=AXIS, linewidth=1)
    for thresh in [-0.3, -0.1, 0.1, 0.3]:
        ax.axvline(thresh, color=GRIDLINE, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("표준화평균차(SMD) — 매칭 전, 처치 대비 대조")
    ax.set_title(f"캠페인 {TARGET_CAMPAIGN} 발행 전 특성 균형 (매칭 전)\n"
                 f"파란색=처치가 더 큼 / 주황색=대조가 더 큼  ·  점선=|SMD| 0.1, 0.3 기준선",
                 fontsize=10, color=INK_PRIMARY, loc="left")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)
    ax.set_axisbelow(True)
    fig.tight_layout()

    smd_path = OUTPUT_DIR / "pretreatment_balance_smd.png"
    fig.savefig(smd_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  저장: {smd_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 5", f"그래프 2 — 분포 비교 (|SMD| >= {SMD_CHART_THRESHOLD}인 {len(large)}개 변수)")
    # ------------------------------------------------------------------
    cols_to_plot = large.sort_values("SMD", key=lambda s: s.abs(), ascending=False)["열"].tolist()
    n = len(cols_to_plot)
    n_col = 3
    n_row = int(np.ceil(n / n_col))

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4.2, n_row * 3.2), dpi=150)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        vals_t, vals_c = t[col].dropna(), c[col].dropna()
        pooled = pd.concat([vals_t, vals_c])
        lo, hi = pooled.quantile(0.01), pooled.quantile(0.99)
        clipped_note = ""
        if pooled.max() > hi:
            clipped_note = " (상위 1% 초과값은 표시 범위 밖)"
        bins = np.linspace(max(lo, 0) if lo >= 0 else lo, hi, 25)

        ax.hist(vals_t.clip(upper=hi), bins=bins, color=COLOR_TREATED, alpha=0.55,
                density=True, label="처치집단", zorder=3)
        ax.hist(vals_c.clip(upper=hi), bins=bins, color=COLOR_CONTROL, alpha=0.55,
                density=True, label="대조집단", zorder=2)
        ax.axvline(vals_t.mean(), color=COLOR_TREATED, linewidth=1.5, linestyle="-", zorder=4)
        ax.axvline(vals_c.mean(), color=COLOR_CONTROL, linewidth=1.5, linestyle="-", zorder=4)

        smd_val = balance.loc[balance["열"] == col, "SMD"].iloc[0]
        ax.set_title(f"{LABELS[col]}{clipped_note}\nSMD={smd_val:.2f}", fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(left=min(lo, 0) if lo < 0 else 0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TREATED, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CONTROL, alpha=0.55),
    ]
    fig.legend(handles, ["처치집단", "대조집단"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10)
    fig.suptitle(f"캠페인 {TARGET_CAMPAIGN} 발행 전 특성 분포 — 매칭 전 (실선=평균)",
                 fontsize=11, color=INK_PRIMARY, y=1.06)
    fig.tight_layout()

    dist_path = OUTPUT_DIR / "pretreatment_balance_dist.png"
    fig.savefig(dist_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {dist_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 6", "요약 — 캠페인 수신 가능성과 관련되어 보이는 특성")
    # ------------------------------------------------------------------
    print("  SMD 절대값 기준 상위 5개:")
    print("  " + balance.dropna(subset=["SMD"]).reindex(
        balance["SMD"].abs().sort_values(ascending=False).index
    ).head(5)[["변수", "SMD"]].to_string(index=False))
    print()
    print(f"  그래프 파일: {smd_path.name}, {dist_path.name}")


if __name__ == "__main__":
    main()
