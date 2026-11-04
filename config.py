"""
config.py  -  ONE place for every path, number and business assumption.
If you want to change a rule (e.g. the cost of a return), change it HERE only.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
DOCS = ROOT / "docs"

SEED = 42                      # makes every random split reproducible
TARGET = "returned"
ID_COL = "order_id"

# ---- Valid ranges (anything outside is a data error, not a real value) ----
RATING_MIN, RATING_MAX = 1.0, 5.0      # star ratings live on a 1-5 scale
ADULT_AGE = 18                         # under-18 buyers are flagged (not deleted)
PRICE_CAP_QUANTILE = 0.999             # extreme prices are capped at this train quantile

# ---- Business assumptions (ASSUMPTIONS - state them in your report!) ----
RETURN_FIXED_COST = 5.0        # reverse shipping + handling per returned order
RETURN_VARIABLE_PCT = 0.15     # restocking / resale loss as % of net order value
HIGH_VALUE_QUANTILE = 0.75     # top 25% orders by net value = High Value
INTERVENE_TOP_PCT = 0.10       # intervene on the top 10% riskiest orders

# ---- Fixed band edges (kept visible so a reader can audit them) ----
AGE_BINS   = [0, 17, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
PRICE_BINS   = [-1, 15, 50, 120, 1e9]
PRICE_LABELS = ["Budget (<15)", "Mid (15-50)", "Premium (50-120)", "Luxury (120+)"]
DISCOUNT_BINS   = [-1, 15, 40, 60, 101]
DISCOUNT_LABELS = ["Low (<15%)", "Medium (15-40%)", "High (40-60%)", "Deep (60%+)"]
RETURN_HIST_BINS   = [-1, 0.10, 0.25, 0.40, 2]
RETURN_HIST_LABELS = ["Low (<10%)", "Medium (10-25%)", "High (25-40%)", "Very High (40%+)"]
DELAY_BINS   = [-100, -1, 1, 3, 100]
DELAY_LABELS = ["Early", "On time", "Late (1-3d)", "Very late (3d+)"]
