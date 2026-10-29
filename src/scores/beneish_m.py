"""
Beneish M-Score implementation for financial statement manipulation detection.

Formula:
The 8-variable Beneish M-Score (Beneish, 1999) predicts the probability of earnings
manipulation using financial statement ratios between consecutive years (t and t-1):

M = -4.84 + 0.920 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI
    + 0.115 * DEPI - 0.172 * SGAI + 4.037 * TATA + 0.0327 * LVGI

Variables:
1. DSRI (Days Sales in Receivables Index):
   (Receivables_t / Sales_t) / (Receivables_{t-1} / Sales_{t-1})
   Measures whether receivables are growing disproportionately to revenue (premature recognition).
2. GMI (Gross Margin Index):
   Gross_Margin_{t-1} / Gross_Margin_t, where Gross_Margin = (Sales - COGS) / Sales
   Measures margin deterioration (> 1 indicates weakening competitive position).
3. AQI (Asset Quality Index):
   [1 - (Current_Assets_t + Net_PPE_t + Securities_t) / Total_Assets_t] /
   [1 - (Current_Assets_{t-1} + Net_PPE_{t-1} + Securities_{t-1}) / Total_Assets_{t-1}]
   Measures non-current asset capitalization of expenses.
4. SGI (Sales Growth Index):
   Sales_t / Sales_{t-1}
   Measures revenue growth momentum (growth firms face pressure to maintain trajectory).
5. DEPI (Depreciation Index):
   Depr_Rate_{t-1} / Depr_Rate_t, where Depr_Rate = Depreciation / (Depreciation + Net_PPE)
   Measures extension of asset depreciable lives (> 1 indicates slowdown in depreciation).
6. SGAI (Sales, General and Administrative expenses Index):
   (SGA_t / Sales_t) / (SGA_{t-1} / Sales_{t-1})
   Measures administrative cost efficiency changes.
7. LVGI (Leverage Index):
   Leverage_t / Leverage_{t-1}, where Leverage = (Current_Liabilities + Long_Term_Debt) / Total_Assets
   Measures increasing balance sheet leverage.
8. TATA (Total Accruals to Total Assets):
   (Net_Income_t - CFO_t) / Total_Assets_t
   Measures non-cash accounting accruals relative to total assets.

Threshold:
- M > -1.78 indicates high likelihood of financial manipulation / accounting distortion.

CRITICAL NOTE ON FINANCIAL INSTITUTIONS:
Beneish M-Score is invalid for Banks and NBFCs because receivables, gross margins,
and standard operating accruals are not reported or structured similarly in lending businesses.
When sector_type="nbfc", this module skips computation and returns NBFC risk proxies.
"""

from typing import Any, Dict, Mapping, Optional
from src.scores.nbfc_proxies import calculate_nbfc_proxies, _get_val


def beneish_m(
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
    sector_type: str = "corporate",
) -> Dict[str, Any]:
    """
    Calculate the 8-variable Beneish M-Score for earnings manipulation detection.

    Parameters
    ----------
    current : Mapping[str, Any]
        Financial metrics for year t (current period).
    prior : Mapping[str, Any]
        Financial metrics for year t-1 (prior period).
    sector_type : str, default 'corporate'
        Company sector category. If 'nbfc' or 'bank', computation is skipped and
        NBFC supervisory proxies are calculated instead.

    Returns
    -------
    Dict[str, Any]
        {
            "score": float or None,
            "is_manipulator": bool or None,
            "variables": {
                "DSRI": float,
                "GMI": float,
                "AQI": float,
                "SGI": float,
                "DEPI": float,
                "SGAI": float,
                "LVGI": float,
                "TATA": float,
            },
            "status": str ("OK", "INVALID_SECTOR", "MISSING_DATA", "MATH_ERROR"),
            "reason": str or None,
            "nbfc_proxies": Dict[str, Any] or None
        }
    """
    # Guard: Sector check for Banks / NBFCs
    if sector_type.strip().lower() in ("nbfc", "bank", "financial", "lender"):
        proxies = calculate_nbfc_proxies(current, prior)
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "INVALID_SECTOR",
            "reason": (
                "Beneish M-Score is not valid for banks and NBFCs (gross margin, COGS, and "
                "receivables indices do not apply to financial lenders). NBFC proxies computed instead."
            ),
            "nbfc_proxies": proxies,
        }

    if prior is None:
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "MISSING_DATA",
            "reason": "Prior year financials (t-1) are required for Beneish M indices",
            "nbfc_proxies": None,
        }

    # Helper to check presence
    missing_fields = []

    def check_req(val: Optional[float], field_name: str) -> Optional[float]:
        if val is None:
            missing_fields.append(field_name)
        return val

    # 1. Sales
    sales_t = check_req(_get_val(current, "sales", "revenue", "turnover", "total_revenue"), "sales_t")
    sales_t1 = check_req(_get_val(prior, "sales", "revenue", "turnover", "total_revenue"), "sales_t1")

    # 2. Receivables
    rec_t = check_req(_get_val(current, "receivables", "trade_receivables", "debtors"), "receivables_t")
    rec_t1 = check_req(_get_val(prior, "receivables", "trade_receivables", "debtors"), "receivables_t1")

    # 3. COGS
    cogs_t = check_req(_get_val(current, "cogs", "cost_of_goods_sold", "cost_of_materials"), "cogs_t")
    cogs_t1 = check_req(_get_val(prior, "cogs", "cost_of_goods_sold", "cost_of_materials"), "cogs_t1")

    # 4. Total Assets
    assets_t = check_req(_get_val(current, "total_assets", "assets"), "total_assets_t")
    assets_t1 = check_req(_get_val(prior, "total_assets", "assets"), "total_assets_t1")

    # 5. Current Assets & Net PPE
    ca_t = check_req(_get_val(current, "current_assets"), "current_assets_t")
    ca_t1 = check_req(_get_val(prior, "current_assets"), "current_assets_t1")

    ppe_t = check_req(_get_val(current, "ppe", "net_ppe", "fixed_assets", "property_plant_equipment"), "ppe_t")
    ppe_t1 = check_req(_get_val(prior, "ppe", "net_ppe", "fixed_assets", "property_plant_equipment"), "ppe_t1")

    # Securities (defaults to 0 if not broken out, per Beneish standard convention)
    sec_t = _get_val(current, "securities", "marketable_securities", "current_investments") or 0.0
    sec_t1 = _get_val(prior, "securities", "marketable_securities", "current_investments") or 0.0

    # 6. Depreciation
    depr_t = check_req(_get_val(current, "depreciation", "depreciation_and_amortization"), "depreciation_t")
    depr_t1 = check_req(_get_val(prior, "depreciation", "depreciation_and_amortization"), "depreciation_t1")

    # 7. SGA Expenses
    sga_t = check_req(_get_val(current, "sga", "sga_expenses", "selling_general_admin"), "sga_t")
    sga_t1 = check_req(_get_val(prior, "sga", "sga_expenses", "selling_general_admin"), "sga_t1")

    # 8. Leverage: Current Liabilities & Long Term Debt (or total liabilities)
    cl_t = _get_val(current, "current_liabilities")
    ltd_t = _get_val(current, "long_term_debt", "non_current_liabilities")
    if cl_t is not None and ltd_t is not None:
        debt_t = cl_t + ltd_t
    else:
        debt_t = _get_val(current, "total_liabilities", "total_debt")
    if debt_t is None:
        missing_fields.append("liabilities_t")

    cl_t1 = _get_val(prior, "current_liabilities")
    ltd_t1 = _get_val(prior, "long_term_debt", "non_current_liabilities")
    if cl_t1 is not None and ltd_t1 is not None:
        debt_t1 = cl_t1 + ltd_t1
    else:
        debt_t1 = _get_val(prior, "total_liabilities", "total_debt")
    if debt_t1 is None:
        missing_fields.append("liabilities_t1")

    # 9. Net Income & CFO (for TATA)
    net_income_t = _get_val(current, "net_income", "pat", "profit_after_tax", "operating_income")
    if net_income_t is None:
        missing_fields.append("net_income_t")

    cfo_t = _get_val(current, "cfo", "cash_flow_from_operations", "operating_cash_flow")
    if cfo_t is None:
        missing_fields.append("cfo_t")

    if missing_fields:
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "MISSING_DATA",
            "reason": f"Missing required fields for Beneish M-Score: {', '.join(missing_fields)}",
            "nbfc_proxies": None,
        }

    # Strict denominator validity checks
    if sales_t <= 0 or sales_t1 <= 0:
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "MATH_ERROR",
            "reason": f"Sales must be positive for both periods, got t={sales_t}, t-1={sales_t1}",
            "nbfc_proxies": None,
        }
    if assets_t <= 0 or assets_t1 <= 0:
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "MATH_ERROR",
            "reason": f"Total assets must be positive for both periods, got t={assets_t}, t-1={assets_t1}",
            "nbfc_proxies": None,
        }

    # 1. DSRI (Days Sales in Receivables Index)
    rec_ratio_t = rec_t / sales_t
    rec_ratio_t1 = rec_t1 / sales_t1
    if rec_ratio_t1 == 0:
        dsri = 1.0 if rec_ratio_t == 0 else 2.0
    else:
        dsri = rec_ratio_t / rec_ratio_t1

    # 2. GMI (Gross Margin Index)
    gm_t = (sales_t - cogs_t) / sales_t
    gm_t1 = (sales_t1 - cogs_t1) / sales_t1
    if gm_t == 0:
        gmi = 1.0 if gm_t1 == 0 else 2.0
    else:
        gmi = gm_t1 / gm_t

    # 3. AQI (Asset Quality Index)
    aq_num = 1.0 - (ca_t + ppe_t + sec_t) / assets_t
    aq_denom = 1.0 - (ca_t1 + ppe_t1 + sec_t1) / assets_t1
    if aq_denom == 0:
        aqi = 1.0 if aq_num == 0 else (1.0 + aq_num)
    else:
        aqi = aq_num / aq_denom

    # 4. SGI (Sales Growth Index)
    sgi = sales_t / sales_t1

    # 5. DEPI (Depreciation Index)
    depr_denom_t = depr_t + ppe_t
    depr_denom_t1 = depr_t1 + ppe_t1
    if depr_denom_t <= 0 or depr_denom_t1 <= 0:
        return {
            "score": None,
            "is_manipulator": None,
            "variables": {},
            "status": "MATH_ERROR",
            "reason": "Depreciation + PPE must be positive for depreciation rate",
            "nbfc_proxies": None,
        }
    depr_rate_t = depr_t / depr_denom_t
    depr_rate_t1 = depr_t1 / depr_denom_t1
    if depr_rate_t == 0:
        depi = 1.0 if depr_rate_t1 == 0 else 2.0
    else:
        depi = depr_rate_t1 / depr_rate_t

    # 6. SGAI (Sales, General and Administrative expenses Index)
    sga_ratio_t = sga_t / sales_t
    sga_ratio_t1 = sga_t1 / sales_t1
    if sga_ratio_t1 == 0:
        sgai = 1.0 if sga_ratio_t == 0 else 2.0
    else:
        sgai = sga_ratio_t / sga_ratio_t1

    # 7. LVGI (Leverage Index)
    lev_t = debt_t / assets_t
    lev_t1 = debt_t1 / assets_t1
    if lev_t1 == 0:
        lvgi = 1.0 if lev_t == 0 else (1.0 + lev_t)
    else:
        lvgi = lev_t / lev_t1

    # 8. TATA (Total Accruals to Total Assets)
    tata = (net_income_t - cfo_t) / assets_t

    # Compute M-Score
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )

    is_manipulator = m_score > -1.78

    return {
        "score": round(m_score, 4),
        "is_manipulator": bool(is_manipulator),
        "variables": {
            "DSRI": round(dsri, 4),
            "GMI": round(gmi, 4),
            "AQI": round(aqi, 4),
            "SGI": round(sgi, 4),
            "DEPI": round(depi, 4),
            "SGAI": round(sgai, 4),
            "LVGI": round(lvgi, 4),
            "TATA": round(tata, 4),
        },
        "threshold": -1.78,
        "status": "OK",
        "reason": None,
        "nbfc_proxies": None,
    }
