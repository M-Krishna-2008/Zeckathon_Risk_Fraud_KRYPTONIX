"""
NBFC and Bank Proxy Metrics for Credit Risk Assessment.

CRITICAL CREDIT POLICY NOTE:
Classical financial distress and manipulation scores (Altman Z, Beneish M, and
Piotroski F) were designed for non-financial industrial and commercial corporate entities.
They are fundamentally INVALID and misleading when applied to Banks and Non-Banking
Financial Companies (NBFCs) for the following structural accounting reasons:
1. Working Capital / Current Ratio: Financial firms borrow short to lend long; deposits
   and short-term borrowings are their operating raw material, not industrial distress debt.
   Working capital is meaningless or perpetually negative for financial intermediaries.
2. Operating Margin / Gross Margin / COGS: NBFCs do not report COGS or inventory. Net Interest
   Margin (NIM) is the core operational spread.
3. Asset Turnover: Lenders possess high asset bases consisting almost entirely of financial
   claims (loans/investments), resulting in structurally low asset turnover that penalizes them
   artificially in manufacturing turnover tests.

When sector_type="nbfc" is encountered, corporate scores are skipped and replaced by
the three primary NBFC supervisory proxies:
- Leverage: Total Debt to Equity (or CRAR / Capital to Risk-Weighted Assets where reported)
- Asset-Quality Trend: Change in Gross Non-Performing Assets ratio (GNPA) or Provisioning
- ALM (Asset-Liability Management) Gap: Structural liquidity mismatch ratio (e.g., Short-Term
  Borrowings / Liquid Assets, or 1-year Cumulative Mismatch / Total Assets).
"""

from typing import Any, Dict, Mapping, Optional


def _is_missing(val: Any) -> bool:
    """Check if a value is None, NaN, or empty."""
    if val is None:
        return True
    try:
        import math
        if isinstance(val, (int, float)) and (math.isnan(val) or math.isinf(val)):
            return True
    except Exception:
        pass
    if isinstance(val, str) and val.strip() in ("", "nan", "NaN", "None", "MISSING"):
        return True
    return False


def _get_val(data: Mapping[str, Any], *keys: str) -> Optional[float]:
    """Extract a float value from mapping using multiple case-insensitive key aliases."""
    for key in keys:
        if key in data and not _is_missing(data[key]):
            try:
                return float(data[key])
            except (ValueError, TypeError):
                return None
        # Try lowercase lookup if key not directly present
        key_lower = key.lower()
        for k, v in data.items():
            if str(k).lower() == key_lower and not _is_missing(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
    return None


def calculate_nbfc_proxies(
    current: Mapping[str, Any],
    prior: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate credit risk proxy metrics tailored for Indian NBFCs and financial lenders.

    Parameters
    ----------
    current : Mapping[str, Any]
        Dictionary or Series of current-period NBFC balance sheet and asset quality items.
    prior : Mapping[str, Any], optional
        Dictionary or Series of prior-period financials for trend analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing available proxy indicators:
        - leverage_ratio: Total Debt / Net Worth (Equity)
        - crar_percent: Capital to Risk-Weighted Assets Ratio (%) if reported
        - gnpa_ratio: Gross NPA / Gross Advances (%)
        - gnpa_trend: Delta in Gross NPA ratio (current - prior)
        - pcr_percent: Provision Coverage Ratio (%)
        - alm_short_term_gap_ratio: Short-Term Borrowings / Liquid Assets (Cash + Bank + G-Secs)
        - status: 'AVAILABLE', 'PARTIAL', or 'MISSING'
        - notes: Explanation for credit officer review.
    """
    proxies: Dict[str, Any] = {
        "sector": "nbfc",
        "leverage_debt_to_equity": None,
        "crar_percent": None,
        "gnpa_ratio": None,
        "gnpa_trend": None,
        "provision_coverage_ratio": None,
        "alm_short_term_gap_ratio": None,
        "available_metrics": [],
        "missing_metrics": [],
        "status": "MISSING",
        "rationale": (
            "NBFC evaluation substituted for industrial scores. Evaluates capital cushion, "
            "asset-quality drift (GNPA), and asset-liability liquidity mismatches."
        ),
    }

    # 1. Leverage & Capital Cushion
    equity = _get_val(current, "equity", "net_worth", "shareholders_equity", "book_value_equity")
    debt = _get_val(current, "total_debt", "borrowings", "total_borrowings", "long_term_debt")
    crar = _get_val(current, "crar", "crar_percent", "car", "capital_adequacy_ratio")

    if debt is not None and equity is not None:
        if equity > 0:
            proxies["leverage_debt_to_equity"] = round(debt / equity, 4)
            proxies["available_metrics"].append("leverage_debt_to_equity")
        else:
            proxies["leverage_debt_to_equity"] = float("inf")
            proxies["available_metrics"].append("leverage_debt_to_equity (negative net worth)")
    else:
        proxies["missing_metrics"].append("leverage_debt_to_equity")

    if crar is not None:
        proxies["crar_percent"] = round(crar, 2)
        proxies["available_metrics"].append("crar_percent")
    else:
        proxies["missing_metrics"].append("crar_percent")

    # 2. Asset Quality & Trend
    gnpa = _get_val(current, "gnpa_ratio", "gnpa_pct", "gross_npa_pct", "gross_npa_ratio")
    if gnpa is None:
        gnpa_val = _get_val(current, "gross_npa", "gnpa_amount")
        advances = _get_val(current, "gross_advances", "total_advances", "loan_book", "aum")
        if gnpa_val is not None and advances is not None and advances > 0:
            gnpa = (gnpa_val / advances) * 100.0

    if gnpa is not None:
        proxies["gnpa_ratio"] = round(gnpa, 4)
        proxies["available_metrics"].append("gnpa_ratio")
        if prior is not None:
            gnpa_prior = _get_val(prior, "gnpa_ratio", "gnpa_pct", "gross_npa_pct", "gross_npa_ratio")
            if gnpa_prior is None:
                gnpa_prior_val = _get_val(prior, "gross_npa", "gnpa_amount")
                adv_prior = _get_val(prior, "gross_advances", "total_advances", "loan_book", "aum")
                if gnpa_prior_val is not None and adv_prior is not None and adv_prior > 0:
                    gnpa_prior = (gnpa_prior_val / adv_prior) * 100.0
            if gnpa_prior is not None:
                proxies["gnpa_trend"] = round(gnpa - gnpa_prior, 4)
                proxies["available_metrics"].append("gnpa_trend")
            else:
                proxies["missing_metrics"].append("gnpa_trend (prior year GNPA missing)")
        else:
            proxies["missing_metrics"].append("gnpa_trend (prior data omitted)")
    else:
        proxies["missing_metrics"].extend(["gnpa_ratio", "gnpa_trend"])

    pcr = _get_val(current, "pcr", "provision_coverage_ratio", "pcr_pct")
    if pcr is not None:
        proxies["provision_coverage_ratio"] = round(pcr, 2)
        proxies["available_metrics"].append("provision_coverage_ratio")
    else:
        proxies["missing_metrics"].append("provision_coverage_ratio")

    # 3. ALM (Asset-Liability Management) Gap Proxy
    short_term_borrowings = _get_val(
        current, "short_term_borrowings", "st_debt", "commercial_paper", "current_liabilities"
    )
    liquid_assets = _get_val(
        current, "liquid_assets", "cash_and_bank", "cash_equivalents", "current_investments"
    )

    if short_term_borrowings is not None and liquid_assets is not None:
        if liquid_assets > 0:
            proxies["alm_short_term_gap_ratio"] = round(short_term_borrowings / liquid_assets, 4)
            proxies["available_metrics"].append("alm_short_term_gap_ratio")
        else:
            proxies["alm_short_term_gap_ratio"] = float("inf")
            proxies["available_metrics"].append("alm_short_term_gap_ratio (zero liquid assets)")
    else:
        proxies["missing_metrics"].append("alm_short_term_gap_ratio")

    if len(proxies["available_metrics"]) > 0:
        proxies["status"] = "AVAILABLE" if len(proxies["missing_metrics"]) == 0 else "PARTIAL"
    else:
        proxies["status"] = "MISSING"

    return proxies
