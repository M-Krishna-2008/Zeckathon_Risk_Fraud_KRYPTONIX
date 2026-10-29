"""
Explainability Module for Early Warning System flags.

Translates complex ML feature attributions (SHAP TreeExplainer), forensic accounting
scores (Beneish M, Altman Z, Piotroski F), and qualitative corporate governance signals
into plain-language rationales for credit risk officers.
"""

from typing import Any, Dict, List, Optional, Tuple


# Template map for Polish ratio features mapped from Indian financials
FEATURE_TEMPLATES = {
    "Attr1": {
        "title": "Return on Assets (ROA)",
        "high_risk": "Negative or collapsing ROA ({val:.1f}%), indicating operating losses eroding book equity.",
        "improving": "ROA recovered to {val:.1f}%, showing positive asset earnings power.",
    },
    "Attr2": {
        "title": "Total Leverage to Assets",
        "high_risk": "Total liabilities elevated at {val:.1f}% of assets, indicating high balance sheet leverage.",
        "improving": "Leverage ratio contained at {val:.1f}% of assets, providing a sound capital buffer.",
    },
    "Attr3": {
        "title": "Working Capital / Total Assets",
        "high_risk": "Working capital ratio deteriorated to {val:.1f}%, reflecting short-term liquidity stress.",
        "improving": "Positive working capital buffer maintained at {val:.1f}% of assets.",
    },
    "Attr4": {
        "title": "Current Ratio",
        "high_risk": "Current ratio fell to {val:.2f}x, indicating insufficient liquid cover for immediate obligations.",
        "improving": "Current ratio healthy at {val:.2f}x, with ample coverage of short-term liabilities.",
    },
    "Attr9": {
        "title": "Sales / Total Assets",
        "high_risk": "Asset turnover slowed to {val:.2f}x, reflecting weakening operational throughput.",
        "improving": "Healthy asset turnover of {val:.2f}x supporting cash generation.",
    },
    "Attr13": {
        "title": "Gross Margin",
        "high_risk": "Gross margin compressed to {val:.1f}%, indicating pricing pressure or surging input costs.",
        "improving": "Stable gross margin at {val:.1f}%, preserving pricing power.",
    },
    "Attr24": {
        "title": "Operating Profit / Assets",
        "high_risk": "EBIT to assets turned negative or depressed at {val:.1f}%, indicating operational strain.",
        "improving": "Operating return on assets solid at {val:.1f}%.",
    },
    "Attr27": {
        "title": "Interest Coverage Ratio (EBIT / Interest)",
        "high_risk": "Interest cover deteriorated to {val:.1f}x, so operating earnings may not cover debt service.",
        "improving": "Comfortable interest cover of {val:.1f}x ensuring debt serviceability.",
    },
    "Attr51": {
        "title": "Short-Term Debt / Assets",
        "high_risk": "Short-term debt rose to {val:.1f}% of total assets, raising rollover and refinancing risk.",
        "improving": "Short-term obligations moderate at {val:.1f}% of total assets.",
    },
    "Attr58": {
        "title": "Operating Costs / Sales",
        "high_risk": "Operating cost ratio reached {val:.1f}% of revenues, leaving little margin for error.",
        "improving": "Operating costs disciplined at {val:.1f}% of revenues.",
    },
}

SOFT_SIGNAL_TEMPLATES = {
    "auditor_resignation": "Auditor abruptly resigned citing '{detail}', signaling acute governance or accounting breakdown.",
    "rating_downgrade": "Credit rating downgraded to '{detail}', indicating formal external assessment of default vulnerability.",
    "related_party_loans": "Related-party transactions flagged ({detail}), raising serious risk of capital diversion.",
    "promoter_pledge_rise": "Promoter share pledge surge reported ({detail}), elevating risk of forced open-market liquidation.",
    "delayed_filing": "Statutory financial results filing delayed ({detail}), typically associated with internal audit disputes.",
}


def explain_flag(
    company: str,
    shap_values: Optional[Dict[str, float]] = None,
    feature_values: Optional[Dict[str, float]] = None,
    scores_summary: Optional[Dict[str, Any]] = None,
    soft_signals: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 3,
) -> List[str]:
    """
    Generate plain-language credit officer risk drivers for a flagged company.

    Combines:
    1. Acute qualitative governance red flags (auditor resignation, rating drop).
    2. Forensic accounting flags (Beneish M, Altman Z, Piotroski F).
    3. Quantitative SHAP tree attribution drivers.

    Returns
    -------
    List[str]
        Top drivers (typically 3) written in concise, credit-committee-ready prose.
    """
    drivers: List[Tuple[float, str]] = []  # (priority_weight, plain_language_text)

    # 1. Soft Signals (Highest priority if present)
    if soft_signals:
        for sig in soft_signals:
            stype = sig.get("signal_type", "")
            detail = sig.get("value", "")
            if stype in SOFT_SIGNAL_TEMPLATES:
                msg = SOFT_SIGNAL_TEMPLATES[stype].format(detail=detail)
                # Soft signals receive high weight in explanation
                weight = 10.0 if stype in ("auditor_resignation", "rating_downgrade") else 7.0
                drivers.append((weight, msg))

    # 2. Forensic & Distress Scores
    if scores_summary:
        beneish = scores_summary.get("beneish_m")
        if beneish and beneish.get("is_manipulator"):
            m_score = beneish.get("score")
            msg = (
                f"Beneish M-Score of {m_score:.2f} (above -1.78 threshold) indicates high probability "
                f"of earnings manipulation or aggressive revenue/accrual distortion."
            )
            drivers.append((8.5, msg))

        altman = scores_summary.get("altman_z")
        if altman and altman.get("zone") == "distress":
            z_score = altman.get("score")
            model_name = "Z''" if altman.get("model_used") == "emerging" else "Z"
            msg = (
                f"Altman {model_name}-Score in distress zone ({z_score:.2f}), "
                f"reflecting severe capital structure and operational insolvency risk."
            )
            drivers.append((8.0, msg))

        piotroski = scores_summary.get("piotroski_f")
        if piotroski and piotroski.get("score") is not None and piotroski.get("score") <= 3:
            f_score = piotroski.get("score")
            failing = ", ".join(piotroski.get("failing_tests", [])[:3])
            msg = (
                f"Piotroski F-Score weak at {f_score}/9, with failures in key fundamentals "
                f"({failing})."
            )
            drivers.append((6.5, msg))

    # 3. Quantitative SHAP feature drivers
    if shap_values and feature_values:
        # Sort SHAP contributions by positive impact on risk probability
        sorted_shap = sorted(
            shap_values.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for feat, val_shap in sorted_shap:
            if val_shap <= 0:
                continue
            if feat in FEATURE_TEMPLATES:
                val = feature_values.get(feat, 0.0)
                template_info = FEATURE_TEMPLATES[feat]
                text = template_info["high_risk"].format(val=val)
                # Weight by SHAP impact
                drivers.append((val_shap * 5.0, text))

    # Sort all drivers by priority/weight descending
    drivers.sort(key=lambda x: x[0], reverse=True)

    # Extract top_n unique drivers
    selected: List[str] = []
    seen = set()
    for _, text in drivers:
        if text not in seen:
            selected.append(text)
            seen.add(text)
            if len(selected) >= top_n:
                break

    # Fallback if no specific drivers were active
    if not selected:
        selected.append(
            f"Borrower {company} exhibits overall composite risk score elevated above the top 5% portfolio cap."
        )

    return selected
