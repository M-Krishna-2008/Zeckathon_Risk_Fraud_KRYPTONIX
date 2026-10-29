"""
Indian corporate distress panel and case data ingestion with point-in-time anti-leakage guards.
"""

from src.india.loader import (
    load_events,
    get_company_event,
    load_peers,
    load_point_in_time_financials,
    load_point_in_time_financials_wide,
    load_point_in_time_soft_signals,
)

__all__ = [
    "load_events",
    "get_company_event",
    "load_peers",
    "load_point_in_time_financials",
    "load_point_in_time_financials_wide",
    "load_point_in_time_soft_signals",
]
