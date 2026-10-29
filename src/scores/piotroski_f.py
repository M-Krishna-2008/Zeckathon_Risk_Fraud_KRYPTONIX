"""
Piotroski F-Score implementation for corporate fundamental financial strength.

Formula & 9 Binary Criteria (Piotroski, 2000):
The F-Score evaluates 9 discrete fundamental signals across 3 dimensions:

Dimension 1: Profitability (Signals 1-4)
  1. F_ROA: Net Income > 0 in current year (1 if positive, else 0).
  2. F_CFO: Cash Flow from Operations > 0 in current year (1 if positive, else 0).
  3. F_DELTA_ROA: ROA_t > ROA_{t-1} (1 if ROA improved, else 0).
  4. F_ACCRUAL: CFO_t > Net_Income_t (1 if cash flow exceeds accounting net income, else 0).

Dimension 2: Leverage, Liquidity, and Source of Funds (Signals 5-7)
  5. F_DELTA_LEVER: Long-Term Debt / Assets in t <= Long-Term Debt / Assets in t-1
     (1 if leverage ratio decreased or stayed 0, else 0).
  6. F_DELTA_LIQUID: Current Ratio in t > Current Ratio in t-1
     (1 if current ratio improved, else 0).
  7. F_EQ_OFFER: Common Shares Outstanding in t <= Common Shares in t-1
     (1 if no dilutive share issuance occurred, else 0).

Dimension 3: Operating Efficiency (Signals 8-9)
  8. F_DELTA_MARGIN: Gross Margin in t > Gross Margin in t-1
     (1 if gross margin improved, else 0).
  9. F_DELTA_TURN: Asset Turnover in t > Asset Turnover in t-1
     (1 if asset turnover improved, else 0).

Aggregate Score:
Sum of all 9 tests (Integer from 0 to 9):
- 0 to 3: Weak / High fundamental distress risk.
- 4 to 6: Average / Moderate stability.
- 7 to 9: Strong fundamental financial health.

CRITICAL NOTE ON FINANCIAL INSTITUTIONS:
Piotroski F-Score criteria (current ratio, gross margin, asset turnover, debt-to-assets)
are meaningless for commercial banks and NBFCs due to their regulatory reserve requirements
and financial intermediation models. When sector_type="nbfc", this module skips computation
and evaluates NBFC risk proxies.
"""

from typing import Any, Dict, List, Mapping, Optional
from src.scores.nbfc_proxies import calculate_nbfc_proxies, _get_val


def piotroski_f(
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
    sector_type: str = "corporate",
) -> Dict[str, Any]:
    """
    Calculate the 9-test Piotroski F-Score with detailed pass/fail audit breakdowns.

    Parameters
    ----------
    current : Mapping[str, Any]
        Financial metrics for year t.
    prior : Mapping[str, Any]
        Financial metrics for year t-1.
    sector_type : str, default 'corporate'
        Company sector category. If 'nbfc' or 'bank', computation is skipped and
        NBFC supervisory proxies are calculated instead.

    Returns
    -------
    Dict[str, Any]
        {
            "score": int or None (0 to 9),
            "strength": str or None ("weak", "moderate", "strong"),
            "failing_tests": List[str],
            "passed_tests": List[str],
            "test_details": Dict[str, Dict[str, Any]],
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
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "INVALID_SECTOR",
            "reason": (
                "Piotroski F-Score is not valid for banks and NBFCs (current ratio, gross margin, "
                "and asset turnover do not reflect banking health). NBFC proxies computed instead."
            ),
            "nbfc_proxies": proxies,
        }

    if prior is None:
        return {
            "score": None,
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "MISSING_DATA",
            "reason": "Prior year financials (t-1) are required for Piotroski F trend tests",
            "nbfc_proxies": None,
        }

    missing_fields = []

    def check_req(val: Optional[float], field_name: str) -> Optional[float]:
        if val is None:
            missing_fields.append(field_name)
        return val

    # Extract required metrics
    assets_t = check_req(_get_val(current, "total_assets", "assets"), "total_assets_t")
    assets_t1 = check_req(_get_val(prior, "total_assets", "assets"), "total_assets_t1")

    net_income_t = check_req(
        _get_val(current, "net_income", "pat", "profit_after_tax"), "net_income_t"
    )
    net_income_t1 = check_req(
        _get_val(prior, "net_income", "pat", "profit_after_tax"), "net_income_t1"
    )

    cfo_t = check_req(_get_val(current, "cfo", "cash_flow_from_operations"), "cfo_t")

    # Long term debt (defaults to 0.0 if explicitly debt-free, but check if key exists)
    ltd_t = _get_val(current, "long_term_debt", "non_current_liabilities", "total_debt")
    ltd_t1 = _get_val(prior, "long_term_debt", "non_current_liabilities", "total_debt")
    if ltd_t is None:
        missing_fields.append("long_term_debt_t")
    if ltd_t1 is None:
        missing_fields.append("long_term_debt_t1")

    # Current ratio components
    ca_t = check_req(_get_val(current, "current_assets"), "current_assets_t")
    cl_t = check_req(_get_val(current, "current_liabilities"), "current_liabilities_t")
    ca_t1 = check_req(_get_val(prior, "current_assets"), "current_assets_t1")
    cl_t1 = check_req(_get_val(prior, "current_liabilities"), "current_liabilities_t1")

    # Shares outstanding
    shares_t = check_req(
        _get_val(current, "shares_outstanding", "common_shares", "num_shares", "share_count"),
        "shares_outstanding_t",
    )
    shares_t1 = check_req(
        _get_val(prior, "shares_outstanding", "common_shares", "num_shares", "share_count"),
        "shares_outstanding_t1",
    )

    # Gross Margin components
    sales_t = check_req(_get_val(current, "sales", "revenue", "turnover"), "sales_t")
    sales_t1 = check_req(_get_val(prior, "sales", "revenue", "turnover"), "sales_t1")

    cogs_t = check_req(_get_val(current, "cogs", "cost_of_goods_sold", "cost_of_materials"), "cogs_t")
    cogs_t1 = check_req(_get_val(prior, "cogs", "cost_of_goods_sold", "cost_of_materials"), "cogs_t1")

    if missing_fields:
        return {
            "score": None,
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "MISSING_DATA",
            "reason": f"Missing required fields for Piotroski F-Score: {', '.join(missing_fields)}",
            "nbfc_proxies": None,
        }

    # Strict denominator validity checks
    if assets_t <= 0 or assets_t1 <= 0:
        return {
            "score": None,
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "MATH_ERROR",
            "reason": f"Total assets must be positive, got t={assets_t}, t-1={assets_t1}",
            "nbfc_proxies": None,
        }
    if cl_t <= 0 or cl_t1 <= 0:
        return {
            "score": None,
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "MATH_ERROR",
            "reason": f"Current liabilities must be positive for current ratio, got t={cl_t}, t-1={cl_t1}",
            "nbfc_proxies": None,
        }
    if sales_t <= 0 or sales_t1 <= 0:
        return {
            "score": None,
            "strength": None,
            "failing_tests": [],
            "passed_tests": [],
            "test_details": {},
            "status": "MATH_ERROR",
            "reason": f"Sales must be positive, got t={sales_t}, t-1={sales_t1}",
            "nbfc_proxies": None,
        }

    test_details: Dict[str, Dict[str, Any]] = {}
    passed_tests: List[str] = []
    failing_tests: List[str] = []

    # 1. F_ROA: Net income > 0
    roa_t = net_income_t / assets_t
    roa_t1 = net_income_t1 / assets_t1
    t1_pass = net_income_t > 0
    test_details["F_ROA"] = {
        "passed": t1_pass,
        "description": "Positive Net Income",
        "value": round(net_income_t, 2),
    }
    (passed_tests if t1_pass else failing_tests).append("F_ROA")

    # 2. F_CFO: Cash Flow from Operations > 0
    t2_pass = cfo_t > 0
    test_details["F_CFO"] = {
        "passed": t2_pass,
        "description": "Positive Operating Cash Flow",
        "value": round(cfo_t, 2),
    }
    (passed_tests if t2_pass else failing_tests).append("F_CFO")

    # 3. F_DELTA_ROA: ROA_t > ROA_{t-1}
    t3_pass = roa_t > roa_t1
    test_details["F_DELTA_ROA"] = {
        "passed": t3_pass,
        "description": "Improving Return on Assets",
        "value_t": round(roa_t, 4),
        "value_t1": round(roa_t1, 4),
    }
    (passed_tests if t3_pass else failing_tests).append("F_DELTA_ROA")

    # 4. F_ACCRUAL: CFO_t > Net_Income_t (Accruals quality)
    t4_pass = cfo_t > net_income_t
    test_details["F_ACCRUAL"] = {
        "passed": t4_pass,
        "description": "CFO Exceeds Net Income (High Accrual Quality)",
        "cfo": round(cfo_t, 2),
        "net_income": round(net_income_t, 2),
    }
    (passed_tests if t4_pass else failing_tests).append("F_ACCRUAL")

    # 5. F_DELTA_LEVER: LTD/Assets decreased or unchanged
    lev_t = ltd_t / assets_t
    lev_t1 = ltd_t1 / assets_t1
    t5_pass = lev_t <= lev_t1
    test_details["F_DELTA_LEVER"] = {
        "passed": t5_pass,
        "description": "Deleveraging / Decreased Long-Term Debt Ratio",
        "leverage_t": round(lev_t, 4),
        "leverage_t1": round(lev_t1, 4),
    }
    (passed_tests if t5_pass else failing_tests).append("F_DELTA_LEVER")

    # 6. F_DELTA_LIQUID: Current Ratio improved
    cr_t = ca_t / cl_t
    cr_t1 = ca_t1 / cl_t1
    t6_pass = cr_t > cr_t1
    test_details["F_DELTA_LIQUID"] = {
        "passed": t6_pass,
        "description": "Improving Current Ratio (Liquidity)",
        "current_ratio_t": round(cr_t, 4),
        "current_ratio_t1": round(cr_t1, 4),
    }
    (passed_tests if t6_pass else failing_tests).append("F_DELTA_LIQUID")

    # 7. F_EQ_OFFER: No new shares issued (shares_t <= shares_t1)
    t7_pass = shares_t <= shares_t1
    test_details["F_EQ_OFFER"] = {
        "passed": t7_pass,
        "description": "No Dilutive Equity Issuance",
        "shares_t": shares_t,
        "shares_t1": shares_t1,
    }
    (passed_tests if t7_pass else failing_tests).append("F_EQ_OFFER")

    # 8. F_DELTA_MARGIN: Gross Margin improved
    gm_t = (sales_t - cogs_t) / sales_t
    gm_t1 = (sales_t1 - cogs_t1) / sales_t1
    t8_pass = gm_t > gm_t1
    test_details["F_DELTA_MARGIN"] = {
        "passed": t8_pass,
        "description": "Improving Gross Margin",
        "gross_margin_t": round(gm_t, 4),
        "gross_margin_t1": round(gm_t1, 4),
    }
    (passed_tests if t8_pass else failing_tests).append("F_DELTA_MARGIN")

    # 9. F_DELTA_TURN: Asset Turnover improved
    turn_t = sales_t / assets_t
    turn_t1 = sales_t1 / assets_t1
    t9_pass = turn_t > turn_t1
    test_details["F_DELTA_TURN"] = {
        "passed": t9_pass,
        "description": "Improving Asset Turnover Efficiency",
        "turnover_t": round(turn_t, 4),
        "turnover_t1": round(turn_t1, 4),
    }
    (passed_tests if t9_pass else failing_tests).append("F_DELTA_TURN")

    score = len(passed_tests)

    # Classify strength
    if score <= 3:
        strength = "weak"
    elif score <= 6:
        strength = "moderate"
    else:
        strength = "strong"

    return {
        "score": score,
        "strength": strength,
        "failing_tests": failing_tests,
        "passed_tests": passed_tests,
        "test_details": test_details,
        "status": "OK",
        "reason": None,
        "nbfc_proxies": None,
    }
