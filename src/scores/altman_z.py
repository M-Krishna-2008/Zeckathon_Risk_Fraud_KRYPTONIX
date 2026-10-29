"""
Altman Z-Score and Altman Z''-Score implementation for corporate credit risk.

Formulas:
1. Original Altman Z-Score (Manufacturing, Public Firms, 1968):
   Z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 0.999 * X5
   Where:
     X1 = Working Capital / Total Assets
     X2 = Retained Earnings / Total Assets
     X3 = EBIT / Total Assets
     X4 = Market Value of Equity / Total Liabilities
     X5 = Sales / Total Assets
   Zones:
     - Distress: Z < 1.81
     - Grey:     1.81 <= Z <= 2.99
     - Safe:     Z > 2.99

2. Altman Z''-Score (Non-Manufacturers & Emerging Markets, 1995/2000):
   Z'' = 6.56 * X1 + 3.26 * X2 + 6.72 * X3 + 1.05 * X4
   Where:
     X1 = Working Capital / Total Assets
     X2 = Retained Earnings / Total Assets
     X3 = EBIT / Total Assets
     X4 = Book Value of Equity / Total Liabilities
   Zones:
     - Distress: Z'' < 1.10
     - Grey:     1.10 <= Z'' <= 2.60
     - Safe:     Z'' > 2.60

CRITICAL NOTE ON FINANCIAL INSTITUTIONS:
Altman Z and Z'' models are mathematically and economically INVALID for Banks and NBFCs.
Working capital, turnover, and asset leverage have completely different operational meanings
in financial intermediation. When sector_type="nbfc", this module skips computation and
evaluates NBFC risk proxies (leverage, asset-quality trend, ALM gap).
"""

from typing import Any, Dict, Mapping, Optional
from src.scores.nbfc_proxies import calculate_nbfc_proxies, _get_val, _is_missing


def _classify_z_zone(score: float, model: str = "manufacturing") -> str:
    """Classify score into distress, grey, or safe zone."""
    if model == "manufacturing":
        if score < 1.81:
            return "distress"
        elif score <= 2.99:
            return "grey"
        else:
            return "safe"
    else:  # emerging / non-manufacturing Z''
        if score < 1.10:
            return "distress"
        elif score <= 2.60:
            return "grey"
        else:
            return "safe"


def altman_z(
    current: Mapping[str, Any],
    prior: Optional[Mapping[str, Any]] = None,
    sector_type: str = "corporate",
    model_type: str = "auto",
) -> Dict[str, Any]:
    """
    Calculate Altman Z-Score (manufacturers) and Z''-Score (emerging markets / non-manufacturers).

    Parameters
    ----------
    current : Mapping[str, Any]
        Dictionary or pandas Series containing current-year balance sheet & P&L metrics:
        Required for Z (manufacturing):
          - 'total_assets' (or 'assets')
          - 'working_capital' (or 'current_assets' and 'current_liabilities')
          - 'retained_earnings' (or 'reserves_and_surplus')
          - 'ebit' (or 'operating_profit')
          - 'market_value_equity' (or 'market_cap')
          - 'total_liabilities' (or 'total_debt' + 'current_liabilities')
          - 'sales' (or 'revenue')
        Required for Z'' (emerging / non-manufacturing):
          - 'total_assets'
          - 'working_capital'
          - 'retained_earnings'
          - 'ebit'
          - 'book_value_equity' (or 'net_worth' or 'equity')
          - 'total_liabilities'
    prior : Mapping[str, Any], optional
        Prior year financials, used if trend or NBFC proxy calculations are requested.
    sector_type : str, default 'corporate'
        Company sector category. If 'nbfc' or 'bank', skips Altman calculation and
        returns NBFC supervisory proxies.
    model_type : str, default 'auto'
        One of 'auto', 'manufacturing', or 'emerging' (Z'').
        If 'auto', calculates both if sufficient data is provided, selecting the
        most appropriate score for the primary 'score' and 'zone' return fields.

    Returns
    -------
    Dict[str, Any]
        {
            "score": float or None,
            "zone": str or None ("distress", "grey", "safe"),
            "model_used": str,
            "z_original": float or None,
            "z_original_zone": str or None,
            "z_double_prime": float or None,
            "z_double_prime_zone": str or None,
            "ratios": Dict[str, float],
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
            "zone": None,
            "model_used": "none",
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "INVALID_SECTOR",
            "reason": (
                "Altman Z-Scores are not valid for banks and NBFCs due to non-comparable "
                "balance-sheet structures (working capital and industrial asset turnover do not apply). "
                "NBFC proxies computed instead."
            ),
            "nbfc_proxies": proxies,
        }

    # Extract common variables
    total_assets = _get_val(current, "total_assets", "assets", "totalAssets")
    if total_assets is None:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MISSING_DATA",
            "reason": "Missing required field: total_assets",
            "nbfc_proxies": None,
        }
    if total_assets <= 0:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MATH_ERROR",
            "reason": f"total_assets must be positive, got {total_assets}",
            "nbfc_proxies": None,
        }

    # Working capital
    working_capital = _get_val(current, "working_capital", "workingCapital")
    if working_capital is None:
        ca = _get_val(current, "current_assets", "currentAssets")
        cl = _get_val(current, "current_liabilities", "currentLiabilities")
        if ca is not None and cl is not None:
            working_capital = ca - cl
        else:
            return {
                "score": None,
                "zone": None,
                "model_used": model_type,
                "z_original": None,
                "z_original_zone": None,
                "z_double_prime": None,
                "z_double_prime_zone": None,
                "ratios": {},
                "status": "MISSING_DATA",
                "reason": "Missing required field: working_capital (or current_assets/current_liabilities)",
                "nbfc_proxies": None,
            }

    # Retained earnings (or reserves_and_surplus / equity fallback)
    retained_earnings = _get_val(
        current,
        "retained_earnings",
        "retainedEarnings",
        "reserves_and_surplus",
        "reserves",
        "equity",
        "net_worth",
    )
    if retained_earnings is None:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MISSING_DATA",
            "reason": "Missing required field: retained_earnings (or reserves_and_surplus)",
            "nbfc_proxies": None,
        }

    # EBIT
    ebit = _get_val(current, "ebit", "operating_profit", "pbit", "operating_income")
    if ebit is None:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MISSING_DATA",
            "reason": "Missing required field: ebit (or operating_profit)",
            "nbfc_proxies": None,
        }

    # Total Liabilities
    total_liabilities = _get_val(current, "total_liabilities", "totalLiabilities")
    if total_liabilities is None:
        cl = _get_val(current, "current_liabilities", "currentLiabilities")
        lt_debt = _get_val(current, "long_term_debt", "non_current_liabilities")
        if cl is not None and lt_debt is not None:
            total_liabilities = cl + lt_debt
        else:
            # Fallback to total_assets - book_equity if explicit
            equity_val = _get_val(current, "book_value_equity", "equity", "net_worth")
            if equity_val is not None:
                total_liabilities = total_assets - equity_val

    if total_liabilities is None:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MISSING_DATA",
            "reason": "Missing required field: total_liabilities",
            "nbfc_proxies": None,
        }
    if total_liabilities <= 0:
        return {
            "score": None,
            "zone": None,
            "model_used": model_type,
            "z_original": None,
            "z_original_zone": None,
            "z_double_prime": None,
            "z_double_prime_zone": None,
            "ratios": {},
            "status": "MATH_ERROR",
            "reason": f"total_liabilities must be positive for leverage ratio, got {total_liabilities}",
            "nbfc_proxies": None,
        }

    # Base Ratios X1, X2, X3
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets

    ratios = {
        "x1_working_capital_to_assets": round(x1, 4),
        "x2_retained_earnings_to_assets": round(x2, 4),
        "x3_ebit_to_assets": round(x3, 4),
    }

    # Try Original Z (Manufacturing)
    mve = _get_val(current, "market_value_equity", "market_cap", "marketCap")
    sales = _get_val(current, "sales", "revenue", "turnover", "total_revenue")

    z_orig = None
    z_orig_zone = None
    missing_for_orig = []
    if mve is None:
        missing_for_orig.append("market_value_equity")
    if sales is None:
        missing_for_orig.append("sales")

    if not missing_for_orig:
        x4_orig = mve / total_liabilities
        x5_orig = sales / total_assets
        ratios["x4_mve_to_liabilities"] = round(x4_orig, 4)
        ratios["x5_sales_to_assets"] = round(x5_orig, 4)
        z_orig = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4_orig + 0.999 * x5_orig, 4)
        z_orig_zone = _classify_z_zone(z_orig, "manufacturing")

    # Try Emerging Market / Non-Manufacturing Z''
    bve = _get_val(current, "book_value_equity", "net_worth", "equity", "shareholders_equity")
    if bve is None and mve is not None:
        # Fallback to market value of equity if book value omitted
        bve = mve

    z_double = None
    z_double_zone = None
    missing_for_double = []
    if bve is None:
        missing_for_double.append("book_value_equity")

    if not missing_for_double:
        x4_double = bve / total_liabilities
        ratios["x4_bve_to_liabilities"] = round(x4_double, 4)
        z_double = round(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4_double, 4)
        z_double_zone = _classify_z_zone(z_double, "emerging")

    # Select Primary Return Score based on model_type
    primary_score = None
    primary_zone = None
    used_model = model_type

    if model_type == "manufacturing":
        if z_orig is not None:
            primary_score = z_orig
            primary_zone = z_orig_zone
        else:
            return {
                "score": None,
                "zone": None,
                "model_used": "manufacturing",
                "z_original": None,
                "z_original_zone": None,
                "z_double_prime": z_double,
                "z_double_prime_zone": z_double_zone,
                "ratios": ratios,
                "status": "MISSING_DATA",
                "reason": f"Missing required fields for manufacturing Z: {', '.join(missing_for_orig)}",
                "nbfc_proxies": None,
            }
    elif model_type in ("emerging", "non_manufacturing", "z_double_prime"):
        if z_double is not None:
            primary_score = z_double
            primary_zone = z_double_zone
        else:
            return {
                "score": None,
                "zone": None,
                "model_used": "emerging",
                "z_original": z_orig,
                "z_original_zone": z_orig_zone,
                "z_double_prime": None,
                "z_double_prime_zone": None,
                "ratios": ratios,
                "status": "MISSING_DATA",
                "reason": f"Missing required fields for Z'': {', '.join(missing_for_double)}",
                "nbfc_proxies": None,
            }
    else:  # auto
        if z_orig is not None:
            primary_score = z_orig
            primary_zone = z_orig_zone
            used_model = "manufacturing"
        elif z_double is not None:
            primary_score = z_double
            primary_zone = z_double_zone
            used_model = "emerging"
        else:
            return {
                "score": None,
                "zone": None,
                "model_used": "auto",
                "z_original": None,
                "z_original_zone": None,
                "z_double_prime": None,
                "z_double_prime_zone": None,
                "ratios": ratios,
                "status": "MISSING_DATA",
                "reason": (
                    f"Insufficient data for either model. Missing for Z: {missing_for_orig}; "
                    f"Missing for Z'': {missing_for_double}"
                ),
                "nbfc_proxies": None,
            }

    return {
        "score": primary_score,
        "zone": primary_zone,
        "model_used": used_model,
        "z_original": z_orig,
        "z_original_zone": z_orig_zone,
        "z_double_prime": z_double,
        "z_double_prime_zone": z_double_zone,
        "ratios": ratios,
        "status": "OK",
        "reason": None,
        "nbfc_proxies": None,
    }
