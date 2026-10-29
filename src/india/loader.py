"""
Indian Corporate Distress Panel Data Loader with Strict Point-in-Time Anti-Leakage Enforcement.

HARD RULE ENFORCEMENT:
Features may use only information publicly available BEFORE the observation / event date.
- Point-in-time filing_date / signal_date is used, NEVER fiscal year-end date.
- Strict condition: record_date < as_of_date. Any record with record_date >= as_of_date
  is strictly discarded.
- Post-event records are NEVER returned when evaluating a company as of an event horizon.
"""

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default paths relative to workspace root
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "india"


def _parse_date(d: Union[str, date, datetime]) -> date:
    """Parse a date string or object into a date instance."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d).strip(), "%Y-%m-%d").date()


def load_events(data_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, str]]:
    """Load verified default and restructuring events."""
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    events_file = folder / "events.csv"
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found at {events_file}")

    records = []
    with open(events_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({k.strip(): v.strip() for k, v in row.items()})
    return records


def get_company_event(
    company: str, data_dir: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, str]]:
    """Retrieve default event metadata for a specific company."""
    company_clean = company.strip().lower()
    for ev in load_events(data_dir):
        if ev["company"].strip().lower() == company_clean:
            return ev
    return None


def load_peers(
    defaulter: Optional[str] = None, data_dir: Optional[Union[str, Path]] = None
) -> List[Dict[str, str]]:
    """Load sector-matched healthy peers."""
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    peers_file = folder / "peers.csv"
    if not peers_file.exists():
        raise FileNotFoundError(f"Peers file not found at {peers_file}")

    records = []
    defaulter_clean = defaulter.strip().lower() if defaulter else None
    with open(peers_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k is not None}
            if defaulter_clean:
                if item.get("defaulter_company", "").strip().lower() == defaulter_clean:
                    records.append(item)
            else:
                records.append(item)
    return records


def load_point_in_time_financials(
    company: str,
    as_of_date: Union[str, date, datetime],
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Load raw long-format financials strictly published before as_of_date.

    Parameters
    ----------
    company : str
        Name of the target company.
    as_of_date : Union[str, date, datetime]
        Observation cutoff date. Only rows with filing_date < as_of_date are returned.
    data_dir : Path, optional
        Data directory location.

    Returns
    -------
    List[Dict[str, Any]]
        List of long-format records matching the point-in-time filter.
    """
    cutoff = _parse_date(as_of_date)
    company_clean = company.strip().lower()
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    fin_file = folder / "financials.csv"

    if not fin_file.exists():
        raise FileNotFoundError(f"Financials file not found at {fin_file}")

    valid_rows = []
    with open(fin_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["company"].strip().lower() != company_clean:
                continue

            # Anti-leakage guard: filing_date strictly BEFORE as_of_date
            filing_str = row["filing_date"].strip()
            filing_d = _parse_date(filing_str)

            if filing_d >= cutoff:
                # STRICT LEAKAGE PREVENTION: Discard anything on or after cutoff
                continue

            try:
                val = float(row["value"].strip())
            except (ValueError, TypeError):
                val = None

            valid_rows.append({
                "company": row["company"].strip(),
                "fiscal_year": row["fiscal_year"].strip(),
                "filing_date": filing_str,
                "item": row["item"].strip(),
                "value": val,
                "source_url": row["source_url"].strip(),
            })

    return valid_rows


def load_point_in_time_financials_wide(
    company: str,
    as_of_date: Union[str, date, datetime],
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pivot long-format point-in-time financials into fiscal_year-keyed dictionaries.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of fiscal_year -> { item_name: value, '_filing_date': str, '_source_urls': list }
    """
    long_rows = load_point_in_time_financials(company, as_of_date, data_dir)
    wide: Dict[str, Dict[str, Any]] = {}

    for r in long_rows:
        fy = r["fiscal_year"]
        if fy not in wide:
            wide[fy] = {
                "_filing_date": r["filing_date"],
                "_sources": set(),
            }
        wide[fy][r["item"]] = r["value"]
        wide[fy]["_sources"].add(r["source_url"])

    # Clean up set to sorted list for serializability
    for fy in wide:
        wide[fy]["_sources"] = sorted(list(wide[fy]["_sources"]))

    return wide


def load_point_in_time_soft_signals(
    company: str,
    as_of_date: Union[str, date, datetime],
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Load qualitative/governance soft signals announced strictly before as_of_date.

    Parameters
    ----------
    company : str
        Company name.
    as_of_date : Union[str, date, datetime]
        Observation cutoff date. Only rows with signal_date < as_of_date are returned.
    data_dir : Path, optional
        Data directory.

    Returns
    -------
    List[Dict[str, Any]]
        List of soft signal records published prior to as_of_date.
    """
    cutoff = _parse_date(as_of_date)
    company_clean = company.strip().lower()
    folder = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sig_file = folder / "soft_signals.csv"

    if not sig_file.exists():
        raise FileNotFoundError(f"Soft signals file not found at {sig_file}")

    valid_signals = []
    with open(sig_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["company"].strip().lower() != company_clean:
                continue

            sig_str = row["signal_date"].strip()
            sig_d = _parse_date(sig_str)

            # Strict leakage guard
            if sig_d >= cutoff:
                continue

            valid_signals.append({
                "company": row["company"].strip(),
                "signal_type": row["signal_type"].strip(),
                "value": row["value"].strip(),
                "signal_date": sig_str,
                "source_url": row["source_url"].strip(),
            })

    return valid_signals
