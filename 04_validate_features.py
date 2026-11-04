"""
PHASE 2B - DID THE FEATURES HELP?  (quick honesty check + data dictionary)
We train the SAME simple models on (a) original columns only vs (b) original + engineered columns,
and compare ROC-AUC on the VALIDATION split. Holdout is untouched (saved for the final test).
Run:  python src/04_validate_features.py
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import config as C

df = pd.read_csv(C.PROCESSED / "train_features.csv")
tr, va = df[df.split=="train"], df[df.split=="validation"]

orig_num = ["customer_age","product_price","discount_percent","product_rating","past_purchase_count",
            "past_return_rate","delivery_delay_days","session_length_minutes","num_product_views","used_coupon"]
cat_orig = ["device_type","product_category","shipping_method","payment_method"]
eng_num = ["discount_amount","net_order_value","est_return_cost","is_high_value_order","log_price","log_net_value",
           "price_vs_cat_median","deep_discount_flag","coupon_deep_discount","discount_vs_cat_mean","serial_returner_flag",
           "expected_past_returns","is_new_customer","is_late","express_late_flag","views_per_minute","low_browse_flag",
           "quick_session_flag","impulse_buy_flag","anomaly_count","price_imputed_flag","discount_neg_flag",
           "rating_clipped_flag","return_rate_neg_flag","session_neg_flag","views_neg_flag","underage_flag"]
eng_cat = ["price_band","discount_band","age_group","return_history_band","customer_segment","delivery_status"]

def run(num, cat, name):
    pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    out = {}
    for mname, m in [("LogisticRegression", LogisticRegression(max_iter=1000)),
                     ("HistGradientBoosting", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=C.SEED))]:
        pipe = Pipeline([("pre", pre), ("m", m)]).fit(tr[num+cat], tr[C.TARGET])
        out[mname] = roc_auc_score(va[C.TARGET], pipe.predict_proba(va[num+cat])[:,1])
    return {"feature_set": name, **out}

res = pd.DataFrame([run(orig_num, cat_orig, "A) Original cleaned columns"),
                    run(orig_num+eng_num, cat_orig+eng_cat, "B) Original + engineered")]).round(4)
res.to_csv(C.REPORTS / "feature_validation_auc.csv", index=False)
print(res.to_string(index=False))

# ---- return rate by key engineered bands (the stories your EDA will tell) ----
fig, axes = plt.subplots(2,3, figsize=(15,8))
for ax, c in zip(axes.ravel(), ["customer_segment","return_history_band","discount_band","delivery_status","price_band","product_category"]):
    g = df[df.split!="test_unlabeled"].groupby(c)[C.TARGET].mean().sort_values()
    g.plot.barh(ax=ax, color="#3D5A80"); ax.axvline(df[C.TARGET].mean(), color="red", ls="--", lw=1)
    ax.set_title(f"Return rate by {c}"); ax.set_xlim(0.40, max(0.56, g.max()+0.01)); ax.set_ylabel("")
plt.suptitle("Return rate by engineered segment (red dashed = overall 47.5%)"); plt.tight_layout()
plt.savefig(C.FIGS / "03_return_rate_by_engineered_segments.png", dpi=110); plt.close()

# ---- data dictionary ----
dd = [
("order_id","ID","Original","Unique order identifier (never used as a model feature)"),
("returned","Target","Original","1 = order was returned, 0 = kept"),
("split","Admin","Added","train/validation/holdout (70/15/15 stratified) or test_unlabeled"),
("customer_age","Numeric","Original","Customer age in years"),
("product_price","Numeric","Cleaned","List price; non-positive values replaced by category median, capped at p99.9"),
("discount_percent","Numeric","Cleaned","Discount 0-100 (negatives set to 0)"),
("product_rating","Numeric","Cleaned","Product star rating clipped to 1-5"),
("past_purchase_count","Numeric","Original","Number of previous orders by the customer"),
("past_return_rate","Numeric","Cleaned","Share of past orders returned, 0-1"),
("delivery_delay_days","Numeric","Original","Days late vs promise; negative = delivered early (valid)"),
("session_length_minutes","Numeric","Cleaned","Browsing session length, >=0"),
("num_product_views","Numeric","Cleaned","Products viewed in session, >=0"),
("device_type","Categorical","Original","desktop / mobile / tablet"),
("product_category","Categorical","Original","beauty, clothing, electronics, home, sports, toys"),
("shipping_method","Categorical","Original","standard / express / same_day"),
("payment_method","Categorical","Original","apple_pay, credit_card, debit_card, paypal"),
("used_coupon","Binary","Original","1 if a coupon was used"),
("price_imputed_flag","Flag","Added","1 if the original price was invalid and was imputed"),
("discount_neg_flag","Flag","Added","1 if original discount was negative"),
("rating_clipped_flag","Flag","Added","1 if original rating was outside 1-5"),
("return_rate_neg_flag","Flag","Added","1 if original return rate was negative"),
("session_neg_flag","Flag","Added","1 if original session length was negative"),
("views_neg_flag","Flag","Added","1 if original view count was negative"),
("underage_flag","Flag","Added","1 if customer_age < 18 (kept, but flagged)"),
("discount_amount","Numeric","Engineered","price x discount% (money given away)"),
("net_order_value","Numeric","Engineered","price - discount_amount = what customer pays (basis of Revenue at Risk)"),
("est_return_cost","Numeric","Engineered","Assumed cost if returned: fixed handling + % of net value (see config.py)"),
("is_high_value_order","Binary","Engineered","1 if net value in top 25% of train orders (matrix row)"),
("price_band","Categorical","Engineered","Budget / Mid / Premium / Luxury"),
("log_price","Numeric","Engineered","log(1+price) to tame skew"),
("log_net_value","Numeric","Engineered","log(1+net value)"),
("price_vs_cat_median","Numeric","Engineered","price / median price of its category"),
("discount_band","Categorical","Engineered","Low / Medium / High / Deep discount"),
("deep_discount_flag","Binary","Engineered","1 if discount >= 60%"),
("coupon_deep_discount","Binary","Engineered","coupon used AND deep discount (stacked promotions)"),
("discount_vs_cat_mean","Numeric","Engineered","discount minus category average discount"),
("age_group","Categorical","Engineered","<18, 18-24, ... 65+"),
("return_history_band","Categorical","Engineered","Low / Medium / High / Very High past return rate"),
("serial_returner_flag","Binary","Engineered","1 if past_return_rate >= 35%"),
("expected_past_returns","Numeric","Engineered","past_purchase_count x past_return_rate"),
("is_new_customer","Binary","Engineered","1 if <= 6 past purchases"),
("customer_segment","Categorical","Engineered","Serial Returner / New-Unproven / Loyal & Reliable / Regular"),
("delivery_status","Categorical","Engineered","Early / On time / Late / Very late"),
("is_late","Binary","Engineered","1 if delivery more than 1 day late"),
("express_late_flag","Binary","Engineered","paid for express/same-day but delivered late (broken promise)"),
("views_per_minute","Numeric","Engineered","views / (session minutes + 1): browsing intensity"),
("low_browse_flag","Binary","Engineered","1 if <= 2 product views (little comparison)"),
("quick_session_flag","Binary","Engineered","1 if session < 15 min"),
("impulse_buy_flag","Binary","Engineered","low browse AND quick session"),
("anomaly_count","Numeric","Engineered","How many data-quality problems the row had"),
("risk_segment","Analysis only","Engineered","category | customer_segment (for risk matrix, not for model)"),
("value_tier","Analysis only","Engineered","High Value / Low Value (for 2x2 decision matrix)"),
]
pd.DataFrame(dd, columns=["column","type","origin","description"]).to_csv(C.DOCS / "data_dictionary.csv", index=False)
print("data dictionary rows:", len(dd))
