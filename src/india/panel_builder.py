"""
Indian Corporate Panel Feature Dataset Builder.

Constructs point-in-time multi-year panel features for all Indian corporate borrowers
combining financial statement ratios, forensic accounting scores (Altman Z, Beneish M, Piotroski F),
mapped Polish bankruptcy benchmark attributes, and governance soft signal indicators.
Outputs to data/india/india_panel.csv.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from src.india.loader import DEFAULT_DATA_DIR, load_events, load_point_in_time_soft_signals
from src.scores.altman_z import altman_z
from src.scores.beneish_m import beneish_m
from src.scores.piotroski_f import piotroski_f
from src.benchmark.mapper import map_indian_to_polish_features
from src.benchmark.model import get_benchmark_model

PANEL_OUTPUT_FILE = DEFAULT_DATA_DIR / "india_panel.csv"


def build_indian_panel(
    data_dir: Optional[Path] = None, output_file: Optional[Path] = None
) -> Path:
    """
    Build and save the consolidated Indian corporate panel dataset.
    """
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    out_path = Path(output_file) if output_file else (folder / "india_panel.csv")
    fin_file = folder / "financials.csv"

    if not fin_file.exists():
        raise FileNotFoundError(f"Financials file not found at {fin_file}")

    # Load raw financials
    df_raw = pd.read_csv(fin_file)
    events = {e["company"].strip().lower(): e for e in load_events(folder)}

    # Group by company and fiscal_year to pivot line items
    companies = df_raw["company"].unique()
    panel_rows: List[Dict[str, Any]] = []

    benchmark_model = get_benchmark_model()

    for comp in sorted(companies):
        c_clean = comp.strip().lower()
        comp_df = df_raw[df_raw["company"] == comp]
        ev = events.get(c_clean)
        sector_type = "nbfc" if (ev and "nbfc" in ev.get("sector", "").lower()) else "corporate"

        # Group by fiscal year
        fys = sorted(comp_df["fiscal_year"].unique())
        fy_dicts: Dict[str, Dict[str, Any]] = {}
        for fy in fys:
            sub = comp_df[comp_df["fiscal_year"] == fy]
            f_date = sub["filing_date"].iloc[0] if "filing_date" in sub.columns else ""
            row_dict = {"_filing_date": f_date}
            for _, r in sub.iterrows():
                try:
                    row_dict[str(r["item"]).strip()] = float(r["value"])
                except (ValueError, TypeError):
                    row_dict[str(r["item"]).strip()] = None
            fy_dicts[fy] = row_dict

        # Process each fiscal year
        for idx, fy in enumerate(fys):
            curr_fin = fy_dicts[fy]
            prior_fin = fy_dicts[fys[idx - 1]] if idx > 0 else None
            filing_date = curr_fin.get("_filing_date", "")

            # 1. Classical and Forensic scores
            z_res = altman_z(curr_fin, prior_fin, sector_type=sector_type)
            m_res = beneish_m(curr_fin, prior_fin, sector_type=sector_type) if prior_fin else None
            f_res = piotroski_f(curr_fin, prior_fin, sector_type=sector_type) if prior_fin else None

            # 2. Mapped Polish Features
            polish_feats = map_indian_to_polish_features(curr_fin)
            base_prob, _ = benchmark_model.predict_probability(polish_feats)

            # 3. Soft signals up to filing date
            soft_signals = []
            if filing_date:
                # Add 1 day to include signals published on filing date
                as_of_d = filing_date
                try:
                    soft_signals = load_point_in_time_soft_signals(comp, as_of_d, data_dir=folder)
                except Exception:
                    soft_signals = []

            # 4. Event outcome
            is_defaulter = 1 if ev else 0
            default_date = ev["event_date"] if ev else ""
            default_type = ev["event_type"] if ev else ""

            row: Dict[str, Any] = {
                "company": comp,
                "fiscal_year": fy,
                "filing_date": filing_date,
                "sector_type": sector_type,
                "is_defaulter": is_defaulter,
                "default_event_date": default_date,
                "default_event_type": default_type,
                "altman_z_score": z_res.get("score"),
                "altman_zone": z_res.get("zone"),
                "beneish_m_score": m_res.get("score") if m_res else None,
                "beneish_is_manipulator": m_res.get("is_manipulator") if m_res else None,
                "piotroski_f_score": f_res.get("score") if f_res else None,
                "benchmark_default_prob": round(base_prob, 4),
                "soft_signals_count": len(soft_signals),
            }

            # Add key reported financial line items
            for item in [
                "total_assets", "equity", "total_liabilities", "current_assets",
                "current_liabilities", "cash_and_equiv", "revenue", "cogs", "ebit",
                "interest_expense", "net_income", "retained_earnings"
            ]:
                row[item] = curr_fin.get(item)

            # Add core Polish Attr features (Attr1..Attr6)
            for i in range(1, 10):
                feat_key = f"Attr{i}"
                row[feat_key] = polish_feats.get(feat_key)

            panel_rows.append(row)

    panel_df = pd.DataFrame(panel_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = build_indian_panel()
    print(f"Indian panel features saved to: {out}")
