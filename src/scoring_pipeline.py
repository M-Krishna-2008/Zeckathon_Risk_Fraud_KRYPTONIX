"""
Quarterly Scoring Pipeline for Indian Corporate Distress.

Combines:
1. Benchmark ML default probability mapped to Polish Attr ratios.
2. Classical & Forensic Accounting Scores (Altman Z/Z'', Beneish M, Piotroski F).
3. Point-in-time Qualitative Soft Signals (Auditor exit, pledge surge, rating downgrade).
4. Transparent log-odds combination layer.
5. Plain-language credit officer rationales (SHAP + Soft Signal synthesis).
"""

from datetime import date, datetime
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from src.india.loader import (
    load_point_in_time_financials_wide,
    load_point_in_time_soft_signals,
    get_company_event,
)
from src.scores.altman_z import altman_z
from src.scores.beneish_m import beneish_m
from src.scores.piotroski_f import piotroski_f
from src.benchmark.mapper import map_indian_to_polish_features
from src.benchmark.model import get_benchmark_model
from src.explain.explainer import explain_flag


def _logit(p: float) -> float:
    p_clipped = max(min(p, 0.999), 0.001)
    return math.log(p_clipped / (1.0 - p_clipped))


def _inv_logit(lodds: float) -> float:
    # Cap to avoid overflow
    lodds_clipped = max(min(lodds, 15.0), -15.0)
    return 1.0 / (1.0 + math.exp(-lodds_clipped))


def score_company(
    company: str,
    as_of_date: Union[str, date, datetime],
    sector_type: str = "corporate",
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Score a single company as of a specific date using strictly point-in-time data.

    Parameters
    ----------
    company : str
        Target company name.
    as_of_date : Union[str, date, datetime]
        Point-in-time observation cutoff date.
    sector_type : str, default 'corporate'
        Sector flag ('corporate' vs 'nbfc').
    data_dir : Path, optional
        Data directory override.

    Returns
    -------
    Dict[str, Any]
        Audit dictionary with composite risk score, probabilities, score components,
        and top plain-language drivers.
    """
    as_of_str = str(as_of_date)[:10]

    # 1. Load point-in-time financials
    wide_fin = load_point_in_time_financials_wide(company, as_of_date, data_dir)
    soft_signals = load_point_in_time_soft_signals(company, as_of_date, data_dir)

    if not wide_fin:
        return {
            "company": company,
            "as_of_date": as_of_str,
            "sector_type": sector_type,
            "status": "NO_PUBLIC_FINANCIALS_PRIOR_TO_CUTOFF",
            "composite_risk_score": 0.04,  # default portfolio prior
            "calibrated_probability": 0.04,
            "is_flagged_5pct_candidate": False,
            "benchmark_probability": None,
            "scores": {},
            "soft_signals": soft_signals,
            "top_drivers": [
                f"No public annual filings available for {company} prior to {as_of_str}."
            ],
        }

    # Sort available fiscal years chronologically
    sorted_fys = sorted(
        wide_fin.keys(),
        key=lambda fy: wide_fin[fy].get("_filing_date", "1900-01-01"),
    )
    latest_fy = sorted_fys[-1]
    prior_fy = sorted_fys[-2] if len(sorted_fys) >= 2 else None

    current_fin = wide_fin[latest_fy]
    prior_fin = wide_fin[prior_fy] if prior_fy else None

    # 2. Compute Accounting & Distress Scores
    altman_res = altman_z(current_fin, prior_fin, sector_type=sector_type)
    beneish_res = beneish_m(current_fin, prior_fin, sector_type=sector_type) if prior_fin else None
    piotroski_res = piotroski_f(current_fin, prior_fin, sector_type=sector_type) if prior_fin else None

    # 3. Benchmark ML Model on Polish Attr Ratios
    polish_features = map_indian_to_polish_features(current_fin)
    benchmark_model = get_benchmark_model()
    base_prob, shap_dict = benchmark_model.predict_probability(polish_features)

    # 4. Transparent Log-Odds Combination Layer
    # Avoid fitting an uncalibrated 50-parameter model on ~10 defaults
    base_lodds = _logit(base_prob)
    delta_lodds = 0.0

    # (a) Accounting Distress Adjustments
    if altman_res.get("zone") == "distress":
        delta_lodds += 0.85
    elif altman_res.get("zone") == "grey":
        delta_lodds += 0.35

    if beneish_res and beneish_res.get("is_manipulator"):
        delta_lodds += 0.95

    if piotroski_res and piotroski_res.get("score") is not None:
        f_score = piotroski_res["score"]
        if f_score <= 3:
            delta_lodds += 0.65
        elif f_score >= 7:
            delta_lodds -= 0.50

    # (b) NBFC & Lender Supervisory Proxy Adjustments
    if sector_type.strip().lower() in ("nbfc", "bank", "financial"):
        proxies = altman_res.get("nbfc_proxies") or {}
        lev = proxies.get("leverage_debt_to_equity")
        if lev is not None and lev >= 7.0:
            delta_lodds += 1.25
        gnpa = proxies.get("gnpa_ratio")
        if gnpa is not None and gnpa >= 5.0:
            delta_lodds += 1.0
        alm = proxies.get("alm_short_term_gap_ratio")
        if alm is not None and alm >= 1.0:
            delta_lodds += 0.9

    # (c) Fundamental Insolvency & Solvency Ratio Adjustments
    ebit_val = current_fin.get("ebit")
    interest_val = current_fin.get("interest_expense")
    net_inc_val = current_fin.get("net_income")
    equity_val = current_fin.get("equity")
    assets_val = current_fin.get("total_assets")

    if equity_val is not None and equity_val < 0:
        delta_lodds += 1.4  # Negative net worth / negative book equity
    if net_inc_val is not None and net_inc_val < 0:
        delta_lodds += 0.7  # Net losses eroding reserves
    if ebit_val is not None and ebit_val < 0:
        delta_lodds += 0.8  # Operating deficit
    if ebit_val is not None and interest_val is not None and interest_val > 0:
        if (ebit_val / interest_val) < 1.0:
            delta_lodds += 0.95  # Operating income does not cover debt interest

    # (d) Soft Signals Adjustments
    for sig in soft_signals:
        st = sig.get("signal_type")
        if st == "rating_downgrade":
            delta_lodds += 2.0  # Formal debt instrument default rating
        elif st == "auditor_resignation":
            delta_lodds += 1.5  # Critical governance red flag
        elif st == "related_party_loans":
            delta_lodds += 0.9
        elif st == "promoter_pledge_rise":
            delta_lodds += 0.7
        elif st == "delayed_filing":
            delta_lodds += 0.4

    composite_prob = round(_inv_logit(base_lodds + delta_lodds), 4)

    # 5. Generate Plain-Language Rationale for Credit Officers
    scores_summary = {
        "altman_z": altman_res,
        "beneish_m": beneish_res,
        "piotroski_f": piotroski_res,
    }
    drivers = explain_flag(
        company=company,
        shap_values=shap_dict,
        feature_values=polish_features,
        scores_summary=scores_summary,
        soft_signals=soft_signals,
        top_n=3,
    )

    return {
        "company": company,
        "as_of_date": as_of_str,
        "latest_filing_used": current_fin.get("_filing_date"),
        "latest_fiscal_year": latest_fy,
        "prior_fiscal_year": prior_fy,
        "sector_type": sector_type,
        "status": "OK",
        "benchmark_base_probability": round(base_prob, 4),
        "composite_risk_score": composite_prob,
        "calibrated_probability": composite_prob,
        "scores": scores_summary,
        "soft_signals": soft_signals,
        "top_drivers": drivers,
        "sources": current_fin.get("_sources", []),
    }
