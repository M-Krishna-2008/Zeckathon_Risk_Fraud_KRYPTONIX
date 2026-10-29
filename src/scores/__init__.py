"""
Financial statement distress, manipulation, and fundamental scoring models.

Models provided:
- altman_z: Classical Altman Z (manufacturers) and Altman Z'' (emerging markets / non-manufacturers).
- beneish_m: 8-variable Beneish M-Score for earnings manipulation detection.
- piotroski_f: 9-test fundamental accounting strength score (0 to 9) with failure breakdown.
- calculate_nbfc_proxies: Specialized proxies for NBFCs & Banks (leverage, GNPA trend, ALM gap).
"""

from src.scores.altman_z import altman_z
from src.scores.beneish_m import beneish_m
from src.scores.piotroski_f import piotroski_f
from src.scores.nbfc_proxies import calculate_nbfc_proxies

__all__ = [
    "altman_z",
    "beneish_m",
    "piotroski_f",
    "calculate_nbfc_proxies",
]
