"""캠페인 18번 분석에서 검증된 처리 규칙 — 모든 캠페인에 동일하게 적용하는 단일 출처.

`pipeline/prepare_data.py`·`pipeline/estimate_effect.py`·`pipeline/run_pipeline.py`가
이 모듈의 상수와 함수만 사용한다. 캠페인마다 규칙이 달라지지 않도록, 규칙을 바꾸려면
이 파일 한 곳만 고치면 되게 만든다(CLAUDE.md 파일 관리 규칙 4).

규칙 출처: analysis/campaign_18_groups.py, campaign_18_pre_features.py,
campaign_18_ps_variables.md, campaign_18_propensity_score.py,
campaign_18_common_support.py, campaign_18_matching.py, campaign_18_effect_estimate.py.
이 규칙들은 캠페인 18 데이터로 검증됐다 — 다른 캠페인에 적용할 때도 "규칙"은 그대로
쓰고, 규칙을 적용한 "결과"(로그변환 대상, 선택된 caliper 등)만 캠페인마다 달라진다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def output_dir(campaign_id: int) -> Path:
    return OUTPUTS_DIR / f"campaign_{campaign_id}"


def save_run_config(campaign_id: int, pre_days: int) -> Path:
    """분석 조건(campaign_id, pre_days)을 기록 — 캐시 재사용 여부 판정에 쓴다."""
    out_dir = output_dir(campaign_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.json"
    config = {
        "campaign_id": int(campaign_id),
        "pre_days": int(pre_days),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def load_run_config(campaign_id: int) -> dict | None:
    path = output_dir(campaign_id) / "run_config.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_cached(campaign_id: int, pre_days: int) -> bool:
    """analysis_data.csv, matched_data.csv, results.json 이 전부 있고, 저장된 조건이
    요청한 campaign_id·pre_days 와 같으면 True — 재계산 없이 재사용 가능하다는 뜻이다."""
    out_dir = output_dir(campaign_id)
    required = ["analysis_data.csv", "results.json"]
    if not all((out_dir / f).exists() for f in required):
        return False
    config = load_run_config(campaign_id)
    if config is None:
        return False
    return config.get("campaign_id") == campaign_id and config.get("pre_days") == pre_days

# =====================================================================
# 1. 집단 구성 규칙 (analysis/campaign_18_groups.py 검증)
# =====================================================================
# 대조집단에 "캠페인을 한 번도 받은 적 없는 가구"를 넣지 않는다. 사용자가 캠페인 18
# 분석에서 명시적으로 결정한 규칙이며, 캠페인 노출 이력이 전혀 없는 가구는 애초에
# 마케팅 프로그램 대상이 아니었을 수 있어 비교 모집단으로 부적절하다고 판단해 고정한다.
INCLUDE_NEVER_CAMPAIGNED = False

# =====================================================================
# 2. 발행 전 특성 계산 규칙 (campaign_18_pre_features.py 검증)
# =====================================================================
# 캠페인 시작일 이전에 '종료'된 캠페인만 pre_campaign_count 에 넣는다. 이 집계는
# pre_days 윈도우와 무관하게 "시작일 이전 전체 이력"을 쓴다(캠페인 18에서 확정한 방식).
ALL_PRE_FEATURE_COLUMNS = [
    "recency", "pre_baskets", "pre_sales", "pre_quantity", "pre_active_days",
    "pre_target_purchase", "pre_target_baskets", "pre_target_sales", "pre_target_quantity",
    "pre_coupon_redemptions", "pre_campaign_count",
]

# 사전 기간에 거래가 전혀 없는 가구의 recency는 "관찰 구간보다 더 오래 안 옴"으로
# 취급해 관찰 구간 길이(pre_days)로 채운다 — NaN으로 두면 모델 입력이 깨진다.
# no_pre_purchase 플래그로 어떤 가구가 이 처리를 받았는지 별도로 남긴다.
RECENCY_NO_PURCHASE_FLAG = "no_pre_purchase"

# =====================================================================
# 3. 성향점수 입력변수 확정 목록 (campaign_18_ps_variables.md 확정)
# =====================================================================
# 제외 사유(캠페인 18에서 검증):
#   pre_active_days      — pre_baskets 와 r=0.97, 중복정보
#   pre_quantity          — QUANTITY 열 단위 혼입(무게 단위 행 존재)으로 신뢰 불가
#   pre_target_purchase   — 캠페인 18에서 분산 0(대상 상품이 카탈로그의 38%라 전원 구매).
#                           캠페인마다 대상 상품 범위가 다르므로 분산이 0이 아닐 수도 있다.
#                           그래도 목록에서는 빼고(캠페인 간 일관성 우선), 분산이 실제로
#                           0인지는 실행 시 검증만 한다(check_variable_variance).
#   pre_target_sales      — pre_sales 와 r=0.94, 중복정보
#   pre_target_quantity   — 같은 이유로 중복, 대상범위가 다른 캠페인에서도 동일 위험
PS_FEATURES = [
    "recency", "pre_baskets", "pre_sales",
    "pre_target_baskets", "pre_coupon_redemptions", "pre_campaign_count",
]

# 인구통계는 완전 배제 — 캠페인 18에서 hh_demographic.csv 커버리지가 37%였고
# 처치·대조 간 26%p 차이가 나 비무작위 결측으로 판단했다. 이 판단은 커버리지 문제이지
# 캠페인 고유 문제가 아니므로(같은 원본 파일을 모든 캠페인이 공유) 전 캠페인에 고정한다.
USE_DEMOGRAPHICS = False

# =====================================================================
# 4. 전처리 규칙 (campaign_18_propensity_score.py 검증)
# =====================================================================
LOG1P_SKEW_THRESHOLD = 1.0   # |왜도| > 1 인 변수에 log1p 적용
STANDARDIZE = True            # 로그변환 후 z-score 표준화, 표본 평균/표준편차 사용

# =====================================================================
# 5. 매칭 규칙 (campaign_18_matching.py 검증)
# =====================================================================
CALIPER_CANDIDATES = [0.1, 0.2, 0.3]   # logit(p_score) 표준편차의 배수
# caliper 선택 규칙(캠페인 18에서 사용자가 확정한 절차):
#   1순위: 매칭 후 최대 |SMD| 최소
#   2순위(동률 또는 근소한 차이일 때): 매칭쌍 수가 더 많은 쪽

# =====================================================================
# 6. 결과변수 규칙 (campaign_18_effect_estimate.py 검증)
# =====================================================================
PRIMARY_COLUMNS = ["target_purchase", "target_sales", "target_quantity"]
SECONDARY_COLUMNS = ["any_purchase", "total_sales", "baskets"]
STANDARD_CAMPAIGN_DAYS = 33   # DATA_DICTIONARY 상 표준 캠페인 기간 근사치
FIRST_N_DAYS = 30             # 기간이 다른 캠페인 비교용 — CLAUDE.md 분석 설계 규칙 6

# =====================================================================
# 7. 품질 게이트 기준값 — CLAUDE.md 품질과 해석 규칙 4
# =====================================================================
MIN_GROUP_SIZE = 30           # 집단 구성 직후 처치·대조 각각의 최소 가구 수
MIN_SUPPORT_SIZE = 30         # 공통지지영역 내 처치·대조 각각의 최소 가구 수
MIN_MATCHED_PAIRS = 30        # 매칭 후 최소 쌍 수
SMD_BALANCE_THRESHOLD = 0.1   # 매칭 후 허용 가능한 최대 |SMD|


# =====================================================================
# 함수 — STEP 1: 캠페인 정보 조회
# =====================================================================
def get_campaign_info(campaign_id: int, campaign_desc: pd.DataFrame | None = None) -> dict:
    """campaign_desc 를 미리 로드해 넘기면(예: Streamlit 캐시) 파일을 다시 읽지 않는다."""
    if campaign_desc is None:
        campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    row = campaign_desc.loc[campaign_desc["CAMPAIGN"] == campaign_id]
    if row.empty:
        raise ValueError(f"campaign_desc.csv에 CAMPAIGN={campaign_id}이 없다.")
    if len(row) > 1:
        raise ValueError(f"campaign_desc.csv에 CAMPAIGN={campaign_id}이 여러 행이다.")
    r = row.iloc[0]
    return {
        "campaign_id": int(campaign_id),
        "description": r["DESCRIPTION"],
        "start_day": int(r["START_DAY"]),
        "end_day": int(r["END_DAY"]),
        "duration": int(r["END_DAY"] - r["START_DAY"] + 1),
    }


# =====================================================================
# 함수 — STEP 2: 집단 구성 (analysis/campaign_18_groups.py 이식)
# =====================================================================
def build_groups(
    campaign_id: int,
    campaign_desc: pd.DataFrame | None = None,
    campaign_table: pd.DataFrame | None = None,
    all_households: set | None = None,
) -> tuple[pd.DataFrame, dict]:
    """세 인자를 미리 로드해 넘기면(예: 대시보드에서 캠페인 여러 개를 반복 조회할 때)
    원본 CSV를 캠페인마다 다시 읽지 않는다. 넘기지 않으면 기존과 동일하게 매번 읽는다."""
    if campaign_desc is None:
        campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    if campaign_table is None:
        campaign_table = pd.read_csv(RAW_DIR / "campaign_table.csv")
    if all_households is None:
        all_households = set(
            pd.read_csv(RAW_DIR / "transaction_data.csv", usecols=["household_key"])["household_key"].unique()
        )

    info = get_campaign_info(campaign_id, campaign_desc)
    start_day, end_day = info["start_day"], info["end_day"]

    others = campaign_desc.loc[campaign_desc["CAMPAIGN"] != campaign_id]
    overlapping_ids = sorted(
        others.loc[(others["START_DAY"] <= end_day) & (start_day <= others["END_DAY"]), "CAMPAIGN"].tolist()
    )

    by_campaign = campaign_table.groupby("CAMPAIGN")["household_key"].apply(set).to_dict()
    campaigned = set(campaign_table["household_key"])
    recipients = by_campaign.get(campaign_id, set())
    overlapping_households: set[int] = set()
    for cid in overlapping_ids:
        overlapping_households |= by_campaign.get(int(cid), set())

    never_campaigned = all_households - campaigned

    treated = recipients - overlapping_households
    excluded_treated_side = recipients & overlapping_households
    excluded_control_side = (campaigned - recipients) & overlapping_households
    control_campaigned = campaigned - recipients - overlapping_households

    if INCLUDE_NEVER_CAMPAIGNED:
        control = control_campaigned | never_campaigned
        excluded_never: set[int] = set()
    else:
        control = control_campaigned
        excluded_never = never_campaigned

    excluded = excluded_treated_side | excluded_control_side | excluded_never

    assignment = pd.DataFrame({"household_key": sorted(all_households)})
    group = pd.Series("제외", index=assignment.index)
    group[assignment["household_key"].isin(treated).values] = "처치"
    group[assignment["household_key"].isin(control).values] = "대조"
    assignment["group"] = group.values
    assignment["treatment"] = (assignment["group"] == "처치").astype(int)
    assignment = assignment[assignment["group"].isin(["처치", "대조"])].reset_index(drop=True)

    meta = {
        **info,
        "overlapping_ids": overlapping_ids,
        "all_households": all_households,
        "campaigned": campaigned,
        "recipients": recipients,
        "overlapping_households": overlapping_households,
        "treated": treated,
        "control": control,
        "excluded": excluded,
        "campaign_table": campaign_table,
        "campaign_desc": campaign_desc,
    }
    return assignment, meta


# =====================================================================
# 함수 — STEP 3: 발행 전 특성 계산 (campaign_18_pre_features.py 이식)
# =====================================================================
def build_pre_features(assignment: pd.DataFrame, meta: dict, pre_days: int) -> tuple[pd.DataFrame, dict]:
    if pre_days <= 0:
        raise ValueError(f"pre_days는 양의 정수여야 한다: {pre_days}")

    start_day, end_day = meta["start_day"], meta["end_day"]
    hh_in_scope = set(assignment["household_key"])

    pre_end = start_day - 1
    pre_start = max(1, start_day - pre_days)
    window_len = pre_end - pre_start + 1

    tx = pd.read_csv(
        RAW_DIR / "transaction_data.csv",
        usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE"],
    )
    tx_scope = tx[tx["household_key"].isin(hh_in_scope)]
    pre_tx = tx_scope[(tx_scope["DAY"] >= pre_start) & (tx_scope["DAY"] <= pre_end)].copy()

    basic = pre_tx.groupby("household_key").agg(
        last_purchase_day=("DAY", "max"),
        pre_active_days=("DAY", "nunique"),
        pre_baskets=("BASKET_ID", "nunique"),
        pre_sales=("SALES_VALUE", "sum"),
        pre_quantity=("QUANTITY", "sum"),
    )
    basic["recency"] = start_day - basic["last_purchase_day"]
    basic = basic.drop(columns=["last_purchase_day"])

    df = assignment.merge(basic, on="household_key", how="left")

    # 결측치 처리 규칙: 사전 기간 거래가 없으면 구매집계는 0(실제로 없었으니까),
    # recency는 "관찰 구간 길이"로 채우고 no_pre_purchase 플래그를 남긴다.
    df[RECENCY_NO_PURCHASE_FLAG] = df["recency"].isna().astype(int)
    for col in ["pre_baskets", "pre_sales", "pre_quantity", "pre_active_days"]:
        df[col] = df[col].fillna(0)
    df["recency"] = df["recency"].fillna(window_len)

    coupon = pd.read_csv(RAW_DIR / "coupon.csv")
    target_products = set(coupon.loc[coupon["CAMPAIGN"] == meta["campaign_id"], "PRODUCT_ID"].unique())
    pre_target_tx = pre_tx[pre_tx["PRODUCT_ID"].isin(target_products)]
    target_agg = pre_target_tx.groupby("household_key").agg(
        pre_target_baskets=("BASKET_ID", "nunique"),
        pre_target_sales=("SALES_VALUE", "sum"),
        pre_target_quantity=("QUANTITY", "sum"),
    )
    df = df.merge(target_agg, on="household_key", how="left")
    for col in ["pre_target_baskets", "pre_target_sales", "pre_target_quantity"]:
        df[col] = df[col].fillna(0)
    df["pre_target_purchase"] = (df["pre_target_baskets"] > 0).astype(int)

    redempt = pd.read_csv(RAW_DIR / "coupon_redempt.csv")
    pre_redempt = redempt[
        redempt["household_key"].isin(hh_in_scope)
        & (redempt["DAY"] >= pre_start) & (redempt["DAY"] <= pre_end)
    ]
    redempt_agg = pre_redempt.groupby("household_key").size().rename("pre_coupon_redemptions")
    df = df.merge(redempt_agg, on="household_key", how="left")
    df["pre_coupon_redemptions"] = df["pre_coupon_redemptions"].fillna(0).astype(int)

    campaign_desc = meta["campaign_desc"]
    campaign_table = meta["campaign_table"]
    pre_campaigns = sorted(campaign_desc.loc[campaign_desc["END_DAY"] < start_day, "CAMPAIGN"].tolist())
    post_campaigns = sorted(campaign_desc.loc[campaign_desc["START_DAY"] > end_day, "CAMPAIGN"].tolist())
    scope_rows = campaign_table[campaign_table["household_key"].isin(hh_in_scope)]
    pre_camp_rows = scope_rows[scope_rows["CAMPAIGN"].isin(pre_campaigns)]
    camp_agg = pre_camp_rows.groupby("household_key").size().rename("pre_campaign_count")
    df = df.merge(camp_agg, on="household_key", how="left")
    df["pre_campaign_count"] = df["pre_campaign_count"].fillna(0).astype(int)

    diag = {
        "pre_start": pre_start, "pre_end": pre_end, "window_len": window_len,
        "pre_tx": pre_tx, "pre_target_tx": pre_target_tx, "pre_redempt": pre_redempt,
        "pre_campaigns": pre_campaigns, "post_campaigns": post_campaigns,
        "target_products": target_products, "tx_scope": tx_scope,
    }
    return df, diag


# =====================================================================
# 함수 — STEP 4: 인코딩 (현재 확정변수는 전부 연속형 — 구조만 유지)
# =====================================================================
def encode_categoricals(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = [c for c in feature_cols if df[c].dtype == object or str(df[c].dtype) == "category"]
    if not cat_cols:
        return df, feature_cols
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    new_cols = [c for c in feature_cols if c not in cat_cols]
    new_cols += [c for c in encoded.columns if any(c.startswith(f"{cc}_") for cc in cat_cols)]
    return encoded, new_cols


def check_variable_variance(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """분산이 0인 변수를 찾는다 — 캠페인마다 대상 상품 범위가 달라 발생할 수 있다."""
    return [c for c in feature_cols if df[c].var(ddof=1) == 0]


# =====================================================================
# 함수 — STEP 5: 결과변수 계산 (campaign_18_build_analysis_table.py 이식)
# =====================================================================
def add_outcome_variables(df: pd.DataFrame, meta: dict, tx_scope: pd.DataFrame, target_products: set) -> pd.DataFrame:
    start_day, end_day = meta["start_day"], meta["end_day"]
    duration = end_day - start_day + 1
    first_end = min(end_day, start_day + FIRST_N_DAYS - 1)

    period_tx = tx_scope[(tx_scope["DAY"] >= start_day) & (tx_scope["DAY"] <= end_day)].copy()
    first30_tx = tx_scope[(tx_scope["DAY"] >= start_day) & (tx_scope["DAY"] <= first_end)].copy()

    def aggregate(period_df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        target_tx = period_df[period_df["PRODUCT_ID"].isin(target_products)]
        primary = target_tx.groupby("household_key").agg(
            target_sales=("SALES_VALUE", "sum"), target_quantity=("QUANTITY", "sum"),
        )
        primary["target_purchase"] = 1
        secondary = period_df.groupby("household_key").agg(
            total_sales=("SALES_VALUE", "sum"), baskets=("BASKET_ID", "nunique"),
        )
        secondary["any_purchase"] = 1
        out = df[["household_key"]].merge(primary, on="household_key", how="left")
        out = out.merge(secondary, on="household_key", how="left")
        for col in ["target_sales", "target_quantity", "total_sales", "baskets"]:
            out[col] = out[col].fillna(0)
        for col in ["target_purchase", "any_purchase"]:
            out[col] = out[col].fillna(0).astype(int)
        if suffix:
            out = out.rename(columns={c: f"{c}{suffix}" for c in out.columns if c != "household_key"})
        return out

    full = aggregate(period_tx)
    first30 = aggregate(first30_tx, suffix="_first30")
    result = df.merge(full, on="household_key", how="left").merge(first30, on="household_key", how="left")
    for col in ["target_sales", "total_sales"]:
        result[f"{col}_per_day"] = (result[col] / duration).round(4)
    return result


# =====================================================================
# 함수 — STEP 6: 성향점수 (campaign_18_propensity_score.py 이식)
# =====================================================================
def fit_propensity_score(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.Series, dict]:
    X = df[feature_cols].copy()
    skew = X.skew()
    log_cols = skew[skew.abs() > LOG1P_SKEW_THRESHOLD].index.tolist()
    for col in log_cols:
        X[col] = np.log1p(X[col])
    X = X.rename(columns={c: f"log1p_{c}" for c in log_cols})

    means, stds = X.mean(), X.std(ddof=0)
    X_std = (X - means) / stds if STANDARDIZE else X

    X_design = sm.add_constant(X_std)
    model = sm.Logit(df["treatment"], X_design)
    result = model.fit(disp=0)
    p_score = result.predict(X_design)

    info = {
        "log_cols": log_cols, "skew": skew, "means": means, "stds": stds,
        "summary": result.summary(), "params": result.params, "pvalues": result.pvalues,
    }
    return p_score, info


# =====================================================================
# 함수 — STEP 7: 공통지지영역
# =====================================================================
def common_support(df: pd.DataFrame) -> tuple[float, float]:
    t = df[df["treatment"] == 1]["p_score"]
    c = df[df["treatment"] == 0]["p_score"]
    return max(t.min(), c.min()), min(t.max(), c.max())


# =====================================================================
# 함수 — STEP 8: 최근접이웃 매칭 (campaign_18_matching.py 이식)
# =====================================================================
def nearest_neighbor_match(pool: pd.DataFrame, caliper: float) -> tuple[list[tuple[int, int]], list[int]]:
    treated = pool[pool["treatment"] == 1].sort_values("household_key")
    control = pool[pool["treatment"] == 0].sort_values("household_key")
    control_available = control.set_index("household_key")["logit_p"].to_dict()
    pairs: list[tuple[int, int]] = []
    unmatched: list[int] = []

    for _, trow in treated.iterrows():
        t_hh, t_logit = int(trow["household_key"]), trow["logit_p"]
        if not control_available:
            unmatched.append(t_hh)
            continue
        candidates = sorted((abs(t_logit - c_logit), c_hh) for c_hh, c_logit in control_available.items())
        best_dist, best_hh = candidates[0]
        if best_dist <= caliper:
            pairs.append((t_hh, best_hh))
            del control_available[best_hh]
        else:
            unmatched.append(t_hh)

    return pairs, unmatched


def compute_smd(df: pd.DataFrame, cols: list[str], pooled_sd: pd.Series) -> pd.Series:
    t = df[df["treatment"] == 1]
    c = df[df["treatment"] == 0]
    diff = t[cols].mean() - c[cols].mean()
    return diff / pooled_sd[cols]


def select_caliper(pool: pd.DataFrame, feature_cols: list[str], pooled_sd: pd.Series) -> dict:
    """caliper 선택 규칙: 매칭 후 최대|SMD| 최소 → 동률/근소차이면 매칭쌍 많은 쪽."""
    logit_sd = pool["logit_p"].std(ddof=1)
    candidates = []
    for m in CALIPER_CANDIDATES:
        caliper = m * logit_sd
        pairs, _ = nearest_neighbor_match(pool, caliper)
        if pairs:
            matched_hh = [h for p in pairs for h in p]
            matched_df = pool[pool["household_key"].isin(matched_hh)]
            smd = compute_smd(matched_df, feature_cols, pooled_sd)
            max_smd = smd.abs().max()
        else:
            max_smd = np.inf
        candidates.append({"multiplier": m, "caliper": caliper, "pairs": pairs, "max_smd": max_smd})

    best = min(candidates, key=lambda c: (round(c["max_smd"], 3), -len(c["pairs"])))
    return {"chosen": best, "all_candidates": candidates, "logit_sd": logit_sd}


# =====================================================================
# 함수 — STEP 9: 효과 추정 CI (campaign_18_effect_estimate.py 이식)
# =====================================================================
def welch_ci(vals_t: pd.Series, vals_c: pd.Series, conf: float = 0.95):
    n_t, n_c = len(vals_t), len(vals_c)
    m_t, m_c = vals_t.mean(), vals_c.mean()
    v_t, v_c = vals_t.var(ddof=1), vals_c.var(ddof=1)
    diff = m_t - m_c
    se = np.sqrt(v_t / n_t + v_c / n_c)
    df = (v_t / n_t + v_c / n_c) ** 2 / ((v_t / n_t) ** 2 / (n_t - 1) + (v_c / n_c) ** 2 / (n_c - 1))
    crit = stats.t.ppf(1 - (1 - conf) / 2, df)
    return diff, se, (diff - crit * se, diff + crit * se), df


def paired_ci(diffs: pd.Series, conf: float = 0.95):
    n = len(diffs)
    mean_d = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    df = n - 1
    crit = stats.t.ppf(1 - (1 - conf) / 2, df)
    return mean_d, se, (mean_d - crit * se, mean_d + crit * se), df
