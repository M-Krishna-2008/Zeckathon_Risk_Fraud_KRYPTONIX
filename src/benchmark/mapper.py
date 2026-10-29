"""
Mapping layer from Indian corporate financial statement items to Polish bankruptcy Attr features.

The benchmark model is based on the 64 classical Polish Bankruptcy attributes (UCI Dataset).
This module maps Indian reported balance-sheet and P&L ratios to the corresponding Attr indices.
Attributes without a direct Indian line-item equivalent remain NaN (missing) so the trained
imputer can process them without guessing or data fabrication.
"""

import math
from typing import Any, Dict, Mapping, Optional


def map_indian_to_polish_features(
    financials: Mapping[str, Any],
) -> Dict[str, float]:
    """
    Map reported Indian financial line items to the 64 Polish bankruptcy Attr ratios.

    Parameters
    ----------
    financials : Mapping[str, Any]
        Dictionary of financial figures for a single fiscal period:
        revenue, net_income, total_assets, total_liabilities, current_assets,
        current_liabilities, working_capital, ebit, interest_expense, cogs, equity.

    Returns
    -------
    Dict[str, float]
        Dictionary keyed by 'Attr1' through 'Attr64'.
        Mapped ratios are floating-point numbers; unmapped attributes are math.nan.
    """
    # Initialize all 64 attributes to NaN
    attrs: Dict[str, float] = {f"Attr{i}": math.nan for i in range(1, 65)}

    def get_num(key: str, fallback_key: Optional[str] = None) -> Optional[float]:
        val = financials.get(key)
        if val is None and fallback_key:
            val = financials.get(fallback_key)
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) or math.isinf(f) else f
        except (ValueError, TypeError):
            return None

    assets = get_num("total_assets", "assets")
    revenue = get_num("revenue", "sales")
    net_income = get_num("net_income", "pat")
    ebit = get_num("ebit", "operating_profit")
    interest = get_num("interest_expense", "finance_costs")
    cogs = get_num("cogs", "cost_of_materials")
    ca = get_num("current_assets")
    cl = get_num("current_liabilities")
    equity = get_num("equity", "net_worth")

    # Working capital
    wc = get_num("working_capital")
    if wc is None and ca is not None and cl is not None:
        wc = ca - cl

    # Total liabilities
    tl = get_num("total_liabilities")
    if tl is None and assets is not None and equity is not None:
        tl = assets - equity

    # Attr1: Net Profit / Total Assets
    if net_income is not None and assets is not None and assets > 0:
        attrs["Attr1"] = round(net_income / assets, 6)

    # Attr2: Total Liabilities / Total Assets
    if tl is not None and assets is not None and assets > 0:
        attrs["Attr2"] = round(tl / assets, 6)

    # Attr3: Working Capital / Total Assets
    if wc is not None and assets is not None and assets > 0:
        attrs["Attr3"] = round(wc / assets, 6)

    # Attr4: Current Assets / Short-Term Liabilities (Current Ratio)
    if ca is not None and cl is not None and cl > 0:
        attrs["Attr4"] = round(ca / cl, 6)

    # Attr9: Sales / Total Assets
    if revenue is not None and assets is not None and assets > 0:
        attrs["Attr9"] = round(revenue / assets, 6)
        attrs["Attr36"] = round(revenue / assets, 6)  # Attr36 is total sales / total assets

    # Attr13: (Gross Profit + Depreciation) / Sales or Gross Margin
    if revenue is not None and cogs is not None and revenue > 0:
        attrs["Attr13"] = round((revenue - cogs) / revenue, 6)

    # Attr24: Gross Profit / Total Assets (or EBIT / Total Assets)
    if ebit is not None and assets is not None and assets > 0:
        attrs["Attr24"] = round(ebit / assets, 6)

    # Attr27: Profit on operating activities / Financial Expenses (Interest Cover)
    if ebit is not None and interest is not None:
        if interest > 0:
            attrs["Attr27"] = round(ebit / interest, 6)
        elif interest == 0 and ebit > 0:
            attrs["Attr27"] = 50.0  # Safe cap for zero-interest debt-free operating profit

    # Attr51: Short-Term Liabilities / Total Assets
    if cl is not None and assets is not None and assets > 0:
        attrs["Attr51"] = round(cl / assets, 6)

    # Attr58: Total Costs / Total Sales = (Revenue - EBIT) / Revenue
    if revenue is not None and ebit is not None and revenue > 0:
        attrs["Attr58"] = round((revenue - ebit) / revenue, 6)

    return attrs
