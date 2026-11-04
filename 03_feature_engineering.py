"""
PHASE 2 - FEATURE ENGINEERING
Turns the cleaned columns into business-meaningful features.
Nothing here looks at the target (returned) -> no data leakage.
Run:  python src/03_feature_engineering.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
import config as C

train = pd.read_csv(C.INTERIM / "train_clean.csv")
test = pd.read_csv(C.INTERIM / "test_clean.csv")

# ---- parameters learned from TRAIN only ----
net_tmp = train.product_price * (1 - train.discount_percent / 100)
P = {
    "high_value_threshold": float(net_tmp.quantile(C.HIGH_VALUE_QUANTILE)),
    "cat_price_median": train.groupby("product_category").product_price.median().to_dict(),
    "cat_discount_mean": train.groupby("product_category").discount_percent.mean().to_dict(),
}

def band(s, bins, labels):
    return pd.cut(s, bins=bins, labels=labels).astype(str)

def add_features(df):
    d = df.copy()
    # --- A. Money features ---
    d["discount_amount"]   = d.product_price * d.discount_percent / 100
    d["net_order_value"]   = d.product_price - d.discount_amount           # what the customer really pays
    d["est_return_cost"]   = C.RETURN_FIXED_COST + C.RETURN_VARIABLE_PCT * d.net_order_value
    d["is_high_value_order"] = (d.net_order_value >= P["high_value_threshold"]).astype(int)
    d["price_band"]        = band(d.product_price, C.PRICE_BINS, C.PRICE_LABELS)
    d["log_price"]         = np.log1p(d.product_price)
    d["log_net_value"]     = np.log1p(d.net_order_value)
    d["price_vs_cat_median"] = d.product_price / d.product_category.map(P["cat_price_median"])

    # --- B. Discount / coupon features ---
    d["discount_band"]     = band(d.discount_percent, C.DISCOUNT_BINS, C.DISCOUNT_LABELS)
    d["deep_discount_flag"] = (d.discount_percent >= 60).astype(int)
    d["coupon_deep_discount"] = d.used_coupon * d.deep_discount_flag
    d["discount_vs_cat_mean"] = d.discount_percent - d.product_category.map(P["cat_discount_mean"])

    # --- C. Customer features ---
    d["age_group"]         = band(d.customer_age, C.AGE_BINS, C.AGE_LABELS)
    d["return_history_band"] = band(d.past_return_rate, C.RETURN_HIST_BINS, C.RETURN_HIST_LABELS)
    d["serial_returner_flag"] = (d.past_return_rate >= 0.35).astype(int)
    d["expected_past_returns"] = d.past_purchase_count * d.past_return_rate   # approx. # of past returned orders
    d["is_new_customer"]   = (d.past_purchase_count <= 6).astype(int)
    d["customer_segment"]  = np.select(
        [d.past_return_rate >= 0.35,
         d.past_purchase_count <= 6,
         (d.past_purchase_count >= 10) & (d.past_return_rate < 0.15)],
        ["Serial Returner", "New / Unproven", "Loyal & Reliable"],
        default="Regular")

    # --- D. Delivery features ---
    d["delivery_status"]   = band(d.delivery_delay_days, C.DELAY_BINS, C.DELAY_LABELS)
    d["is_late"]           = (d.delivery_delay_days > 1).astype(int)
    d["express_late_flag"] = ((d.shipping_method.isin(["express","same_day"])) & (d.delivery_delay_days > 0)).astype(int)

    # --- E. Behaviour features ---
    d["views_per_minute"]  = d.num_product_views / (d.session_length_minutes + 1)
    d["low_browse_flag"]   = (d.num_product_views <= 2).astype(int)        # bought with almost no comparison
    d["quick_session_flag"] = (d.session_length_minutes < 15).astype(int)
    d["impulse_buy_flag"]  = d.low_browse_flag * d.quick_session_flag

    # --- F. Data-quality summary ---
    flag_cols = [c for c in d.columns if c.endswith("_flag") and c in
                 ("price_imputed_flag","discount_neg_flag","rating_clipped_flag","return_rate_neg_flag","session_neg_flag","views_neg_flag")]
    d["anomaly_count"]     = d[flag_cols].sum(axis=1)

    # --- G. Analysis-only keys (for the dashboard risk matrix; NOT for the model) ---
    d["risk_segment"]      = d.product_category + " | " + d.customer_segment
    d["value_tier"]        = np.where(d.is_high_value_order == 1, "High Value", "Low Value")
    return d

tr = add_features(train)
te = add_features(test)

# ---- Train / Validation / Holdout split (70 / 15 / 15), stratified on target ----
idx_train, idx_tmp = train_test_split(tr.index, test_size=0.30, random_state=C.SEED, stratify=tr[C.TARGET])
idx_val, idx_hold = train_test_split(idx_tmp, test_size=0.50, random_state=C.SEED, stratify=tr.loc[idx_tmp, C.TARGET])
tr["split"] = "train"; tr.loc[idx_val, "split"] = "validation"; tr.loc[idx_hold, "split"] = "holdout"
te["split"] = "test_unlabeled"

# column order: id, target, split, then everything else
first = [C.ID_COL, C.TARGET, "split"]
tr = tr[first + [c for c in tr.columns if c not in first]]
te = te[[C.ID_COL, "split"] + [c for c in te.columns if c not in (C.ID_COL, "split")]]

tr.to_csv(C.PROCESSED / "train_features.csv", index=False)
te.to_csv(C.PROCESSED / "test_features.csv", index=False)
(C.PROCESSED / "feature_params.json").write_text(json.dumps(P, indent=2))

print("train_features:", tr.shape, "| test_features:", te.shape)
print(tr.split.value_counts().to_dict())
print(tr.groupby("split")[C.TARGET].mean().round(4).to_dict())
print("\nsegment sizes:\n", tr.customer_segment.value_counts().to_string())
print("\nnew columns added:", [c for c in tr.columns if c not in pd.read_csv(C.RAW/"train.csv", nrows=1).columns])
