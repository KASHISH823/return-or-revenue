"""
PHASE 1A - DATA AUDIT (look, do not touch)
Goal: understand the raw data BEFORE changing anything.
Run:  python src/01_data_audit.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

train = pd.read_csv(C.RAW / "train.csv")
test = pd.read_csv(C.RAW / "test.csv")

lines = []
def log(s=""):
    print(s); lines.append(s)

log("# Raw Data Audit Report\n")
log(f"- train shape: {train.shape}  | test shape: {test.shape}")
log(f"- columns only in train: {sorted(set(train.columns)-set(test.columns))}")
log(f"- missing values train: {int(train.isna().sum().sum())} | test: {int(test.isna().sum().sum())}")
log(f"- fully duplicated rows train: {int(train.duplicated().sum())}")
log(f"- duplicated order_id train: {int(train[C.ID_COL].duplicated().sum())} | test: {int(test[C.ID_COL].duplicated().sum())}")
log(f"- order_id overlap train/test: {len(set(train[C.ID_COL]) & set(test[C.ID_COL]))}")
log(f"\n## Target distribution (returned)\n")
log(train[C.TARGET].value_counts(normalize=True).round(4).to_string())

num_cols = ["customer_age","product_price","discount_percent","product_rating","past_purchase_count",
            "past_return_rate","delivery_delay_days","session_length_minutes","num_product_views"]
log("\n## Impossible / suspicious values (train)\n")
rows = []
rules = {
 "product_price": ("<= 0", train.product_price <= 0),
 "discount_percent": ("< 0", train.discount_percent < 0),
 "product_rating": ("outside 1-5", (train.product_rating < 1) | (train.product_rating > 5)),
 "past_return_rate": ("< 0", train.past_return_rate < 0),
 "session_length_minutes": ("< 0", train.session_length_minutes < 0),
 "num_product_views": ("< 0", train.num_product_views < 0),
 "customer_age": ("< 18", train.customer_age < 18),
}
for col,(rule,mask) in rules.items():
    rows.append({"column":col,"rule":rule,"rows":int(mask.sum()),"pct":round(100*mask.mean(),2)})
audit = pd.DataFrame(rows); log(audit.to_string(index=False))
audit.to_csv(C.REPORTS / "raw_anomaly_summary.csv", index=False)

log("\n## Categorical columns\n")
for c in ["device_type","product_category","shipping_method","payment_method","used_coupon"]:
    log(f"{c}: {sorted(train[c].unique().tolist())}")

log("\n## Train vs Test distribution check (means) - should be similar\n")
cmp = pd.DataFrame({"train_mean": train[num_cols].mean(), "test_mean": test[num_cols].mean()}).round(3)
log(cmp.to_string())

log("\n## Correlation of each numeric column with returned\n")
log(train[num_cols + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET).sort_values().round(4).to_string())

(C.REPORTS / "01_raw_data_audit.md").write_text("\n".join(lines))

# ---- Figures ----
fig, ax = plt.subplots(figsize=(4.5,4))
train[C.TARGET].value_counts().sort_index().plot.bar(ax=ax, color=["#4C9F70","#D1495B"])
ax.set_xticklabels(["Kept (0)","Returned (1)"], rotation=0); ax.set_title("Target balance"); ax.set_ylabel("orders")
plt.tight_layout(); plt.savefig(C.FIGS / "01_target_distribution.png", dpi=130); plt.close()

fig, axes = plt.subplots(3,3, figsize=(13,9))
for ax, c in zip(axes.ravel(), num_cols):
    ax.hist(train[c], bins=50, color="#3D5A80"); ax.set_title(c)
    if c in ("product_price","discount_percent","past_return_rate","session_length_minutes","num_product_views"):
        ax.axvline(0, color="red", lw=1)
plt.suptitle("Raw numeric distributions (red line = zero; mass left of it = invalid values)")
plt.tight_layout(); plt.savefig(C.FIGS / "02_raw_numeric_distributions.png", dpi=110); plt.close()
print("\nSaved report + figures.")
