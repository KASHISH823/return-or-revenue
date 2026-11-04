"""
PHASE 1B - DATA CLEANING
Rule: we never delete rows here. We FLAG bad values, then FIX them with a documented rule.
All statistics (caps, medians) are learned from TRAIN only and re-used on TEST (no leakage).
Run:  python src/02_clean_data.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import json
import pandas as pd, numpy as np
import config as C

train = pd.read_csv(C.RAW / "train.csv")
test = pd.read_csv(C.RAW / "test.csv")

# ---------- 1. learn parameters from TRAIN ----------
valid_price = train[train.product_price > 0]
params = {
    "price_cap": float(valid_price.product_price.quantile(C.PRICE_CAP_QUANTILE)),
    "price_median_by_category": valid_price.groupby("product_category").product_price.median().to_dict(),
    "price_median_overall": float(valid_price.product_price.median()),
}

def clean(df, name):
    df = df.copy()
    log = []
    # text columns: strip + lower-case so "Mobile" and "mobile " are the same
    for c in ["device_type","product_category","shipping_method","payment_method"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # ---- flags FIRST (so the information that a value was wrong is never lost) ----
    df["price_imputed_flag"]      = (df.product_price <= 0).astype(int)
    df["discount_neg_flag"]       = (df.discount_percent < 0).astype(int)
    df["rating_clipped_flag"]     = ((df.product_rating < C.RATING_MIN) | (df.product_rating > C.RATING_MAX)).astype(int)
    df["return_rate_neg_flag"]    = (df.past_return_rate < 0).astype(int)
    df["session_neg_flag"]        = (df.session_length_minutes < 0).astype(int)
    df["views_neg_flag"]          = (df.num_product_views < 0).astype(int)
    df["underage_flag"]           = (df.customer_age < C.ADULT_AGE).astype(int)

    # ---- fixes ----
    n = df.price_imputed_flag.sum()
    cat_med = df.product_category.map(params["price_median_by_category"]).fillna(params["price_median_overall"])
    df["product_price"] = np.where(df.product_price <= 0, cat_med, df.product_price)
    log.append(("product_price", "<=0 replaced by category median of valid prices", int(n)))

    n = (df.product_price > params["price_cap"]).sum()
    df["product_price"] = df.product_price.clip(upper=params["price_cap"])
    log.append(("product_price", f"capped at train p{C.PRICE_CAP_QUANTILE*100:.1f} = {params[chr(112)+chr(114)+chr(105)+chr(99)+chr(101)+chr(95)+chr(99)+chr(97)+chr(112)]:.1f}", int(n)))

    for col, lo, hi, why in [
        ("discount_percent", 0, 100, "negative discount -> 0"),
        ("product_rating", C.RATING_MIN, C.RATING_MAX, "clipped to 1-5 star scale"),
        ("past_return_rate", 0, 1, "negative rate -> 0 (rate must be 0-1)"),
        ("session_length_minutes", 0, None, "negative minutes -> 0"),
        ("num_product_views", 0, None, "negative views -> 0"),
    ]:
        bad = ((df[col] < lo) | ((df[col] > hi) if hi is not None else False)).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        log.append((col, why, int(bad)))

    # age is real-looking; keep but flag (underage_flag). Cast integer columns
    df["num_product_views"] = df.num_product_views.round().astype(int)
    df["customer_age"] = df.customer_age.astype(int)
    df["past_purchase_count"] = df.past_purchase_count.astype(int)
    df["used_coupon"] = df.used_coupon.astype(int)

    # sanity asserts - script stops if something is still wrong
    assert df.isna().sum().sum() == 0
    assert (df.product_price > 0).all() and df.discount_percent.between(0,100).all()
    assert df.product_rating.between(1,5).all() and df.past_return_rate.between(0,1).all()
    assert (df.session_length_minutes >= 0).all() and (df.num_product_views >= 0).all()
    return df, pd.DataFrame(log, columns=["column","rule","rows_changed"]).assign(dataset=name)

train_c, l1 = clean(train, "train")
test_c, l2 = clean(test, "test")
train_c.to_csv(C.INTERIM / "train_clean.csv", index=False)
test_c.to_csv(C.INTERIM / "test_clean.csv", index=False)
cl = pd.concat([l1, l2]); cl.to_csv(C.REPORTS / "cleaning_log.csv", index=False)
(C.INTERIM / "cleaning_params.json").write_text(json.dumps(params, indent=2))
print(cl.to_string(index=False))
print("\nrows train:", len(train_c), "| rows test:", len(test_c), "(no rows dropped)")
