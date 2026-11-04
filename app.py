import streamlit as st
import pandas as pd
import plotly.express as px

# Load final business risk dataset
final_dataset = pd.read_csv("final_business_risk_dataset.csv")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Return & Revenue Risk Control",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("final_business_risk_dataset.csv")
    return df


df = load_data()

# ============================================================
# TITLE
# ============================================================

st.title("📊 Return & Revenue Risk Control")

st.markdown(
    """
    **ML-powered business risk analysis**

    This dashboard identifies orders with higher return risk,
    estimates financial exposure, and helps prioritize
    business intervention.
    """
)
st.divider()

# ============================================================
# EXECUTIVE KPI HEADER
# ============================================================

total_orders = len(final_dataset)

actual_return_rate = df["returned"].mean()

total_revenue_risk = final_dataset["revenue_at_risk"].sum()

total_expected_loss = final_dataset["expected_return_loss"].sum()

high_critical_orders = final_dataset[
    final_dataset["business_risk_tier"].isin(["High", "Critical"])
].shape[0]

st.markdown("### 📊 Business Risk Control Dashboard")

st.caption(
    "ML-powered return risk prediction, financial exposure analysis "
    "and intervention planning"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

with kpi2:
    st.metric(
        "Return Rate",
        f"{actual_return_rate:.1%}"
    )

with kpi3:
    st.metric(
        "Revenue at Risk",
        f"₹{total_revenue_risk:,.0f}"
    )

with kpi4:
    st.metric(
        "Expected Return Loss",
        f"₹{total_expected_loss:,.0f}"
    )

with kpi5:
    st.metric(
        "High + Critical Orders",
        f"{high_critical_orders:,}"
    )

st.markdown("---")

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown("---")

st.header("📌 Executive Summary")

# Calculate actual return rate from original dataset
actual_return_rate = df["returned"].mean()

st.write(
    f"""
    The analysis covers **{len(final_dataset):,} orders** and identifies
    orders that may create significant financial exposure through
    product returns.

    The overall return rate is approximately **{actual_return_rate:.1%}**.

    The model estimates total **Revenue at Risk of ₹{final_dataset['revenue_at_risk'].sum():,.0f}**
    and an expected return-related loss of
    **₹{final_dataset['expected_return_loss'].sum():,.0f}**.

    The business should prioritize high-risk and high-value orders
    instead of applying the same intervention to every customer.
    """
)

# ============================================================
# DASHBOARD FILTERS
# ============================================================

st.markdown("---")

st.header("🔎 Analyze Specific Business Segments")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    selected_categories = st.multiselect(
        "Product Category",
        options=sorted(final_dataset["product_category"].dropna().unique()),
        default=sorted(final_dataset["product_category"].dropna().unique())
    )

with filter_col2:
    selected_segments = st.multiselect(
        "Customer Segment",
        options=sorted(final_dataset["customer_segment"].dropna().unique()),
        default=sorted(final_dataset["customer_segment"].dropna().unique())
    )

with filter_col3:
    selected_risk = st.multiselect(
        "Business Risk",
        options=sorted(final_dataset["business_risk_tier"].dropna().unique()),
        default=sorted(final_dataset["business_risk_tier"].dropna().unique())
    )

with filter_col4:
    selected_priority = st.multiselect(
        "Priority Tier",
        options=sorted(final_dataset["priority_tier"].dropna().unique()),
        default=sorted(final_dataset["priority_tier"].dropna().unique())
    )

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = final_dataset[
    final_dataset["product_category"].isin(selected_categories)
    & final_dataset["customer_segment"].isin(selected_segments)
    & final_dataset["business_risk_tier"].isin(selected_risk)
    & final_dataset["priority_tier"].isin(selected_priority)
].copy()

# ============================================================
# FILTERED KPIs
# ============================================================

st.subheader("📌 Selected Segment Metrics")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Filtered Orders",
        f"{len(filtered_df):,}"
    )

with kpi2:
    st.metric(
        "Revenue at Risk",
        f"₹{filtered_df['revenue_at_risk'].sum():,.0f}"
    )

with kpi3:
    st.metric(
        "Expected Return Loss",
        f"₹{filtered_df['expected_return_loss'].sum():,.0f}"
    )

with kpi4:
    st.metric(
        "Average Risk",
        f"{filtered_df['risk_prob'].mean():.1%}"
    )

# ============================================================
# BASIC BUSINESS METRICS
# ============================================================

total_orders = len(df)

return_rate = df["returned"].mean()

total_revenue_risk = df["revenue_at_risk"].sum()

total_expected_loss = df["expected_return_loss"].sum()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


with col2:
    st.metric(
        "Return Rate",
        f"{return_rate:.1%}"
    )


with col3:
    st.metric(
        "Revenue at Risk",
        f"₹{total_revenue_risk:,.0f}"
    )


with col4:
    st.metric(
        "Expected Return Loss",
        f"₹{total_expected_loss:,.0f}"
    )


st.divider()


# ============================================================
# BUSINESS RISK DISTRIBUTION
# ============================================================

st.subheader("Business Risk Distribution")

risk_order = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

risk_counts = (
    df["business_risk_tier"]
    .value_counts()
    .reindex(
        risk_order,
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk Tier",
    "Orders"
]

fig_risk = px.bar(
    risk_counts,
    x="Risk Tier",
    y="Orders",
    title="Orders by Business Risk Tier",
    text="Orders"
)

fig_risk.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_risk.update_layout(
    xaxis_title="Business Risk Tier",
    yaxis_title="Number of Orders",
    showlegend=False
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)

# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

st.subheader("Priority Distribution")

priority_counts = (
    df["priority_tier"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High", "Critical"],
        fill_value=0
    )
    .reset_index()
)

priority_counts.columns = [
    "Priority Tier",
    "Orders"
]

fig_priority = px.bar(
    priority_counts,
    x="Priority Tier",
    y="Orders",
    title="Orders by Intervention Priority",
    text="Orders"
)

fig_priority.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_priority.update_layout(
    xaxis_title="Priority Tier",
    yaxis_title="Number of Orders",
    showlegend=False
)

st.plotly_chart(
    fig_priority,
    use_container_width=True
)


# ============================================================
# HIGH-PRIORITY ORDERS
# ============================================================

st.subheader("High-Priority Orders")

priority_orders = (
    df
    .sort_values(
        "financial_exposure",
        ascending=False
    )
    .head(20)
)

display_columns = [
    "order_id",
    "risk_prob",
    "net_order_value",
    "revenue_at_risk",
    "expected_return_loss",
    "priority_tier",
    "business_risk_tier"
]

st.dataframe(
    priority_orders[display_columns],
    use_container_width=True,
    hide_index=True
)

# ============================================================
# PRODUCT CATEGORY ANALYSIS
# ============================================================

st.divider()

st.subheader("Product Category Risk Analysis")

category_summary = (
    df.groupby("product_category")
    .agg(
        orders=("order_id", "count"),
        order_value=("net_order_value", "sum"),
        return_rate=("returned", "mean"),
        revenue_at_risk=("revenue_at_risk", "sum")
    )
    .reset_index()
)

category_summary = category_summary.sort_values(
    "revenue_at_risk",
    ascending=False
)

category_display = category_summary.copy()

category_display["order_value"] = (
    category_display["order_value"].round(2)
)

category_display["return_rate"] = (
    category_display["return_rate"] * 100
).round(1)

category_display["revenue_at_risk"] = (
    category_display["revenue_at_risk"].round(2)
)

category_display.columns = [
    "Product Category",
    "Orders",
    "Order Value",
    "Return Rate (%)",
    "Revenue at Risk"
]

st.dataframe(
    category_display,
    width="stretch",
    hide_index=True
)

# ============================================================
# REVENUE AT RISK BY PRODUCT CATEGORY
# ============================================================

fig_category = px.bar(
    category_summary,
    x="product_category",
    y="revenue_at_risk",
    title="Revenue at Risk by Product Category",
    text="revenue_at_risk"
)

fig_category.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_category.update_layout(
    xaxis_title="Product Category",
    yaxis_title="Revenue at Risk",
    showlegend=False
)

st.plotly_chart(
    fig_category,
    width="stretch"
)

# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

st.divider()

st.subheader("Customer Segment Risk Analysis")

segment_summary = (
    df.groupby("customer_segment")
    .agg(
        orders=("order_id", "count"),
        order_value=("net_order_value", "sum"),
        return_rate=("returned", "mean"),
        revenue_at_risk=("revenue_at_risk", "sum")
    )
    .reset_index()
)

segment_summary = segment_summary.sort_values(
    "revenue_at_risk",
    ascending=False
)

segment_display = segment_summary.copy()

segment_display["order_value"] = (
    segment_display["order_value"].round(2)
)

segment_display["return_rate"] = (
    segment_display["return_rate"] * 100
).round(1)

segment_display["revenue_at_risk"] = (
    segment_display["revenue_at_risk"].round(2)
)

segment_display.columns = [
    "Customer Segment",
    "Orders",
    "Order Value",
    "Return Rate (%)",
    "Revenue at Risk"
]

st.dataframe(
    segment_display,
    width="stretch",
    hide_index=True
)

# ============================================================
# REVENUE AT RISK BY CUSTOMER SEGMENT
# ============================================================

fig_segment = px.bar(
    segment_summary,
    x="customer_segment",
    y="revenue_at_risk",
    title="Revenue at Risk by Customer Segment",
    text="revenue_at_risk"
)

fig_segment.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_segment.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Revenue at Risk",
    showlegend=False
)

st.plotly_chart(
    fig_segment,
    width="stretch"
)

# ============================================================
# TOP FINANCIAL EXPOSURE ORDERS
# ============================================================

st.divider()

st.subheader("🚨 Top Orders by Financial Exposure")

top_exposure = (
    filtered_df
    .sort_values(
        "financial_exposure",
        ascending=False
    )
    .head(20)
)

exposure_columns = [
    "order_id",
    "risk_prob",
    "net_order_value",
    "est_return_cost",
    "financial_exposure",
    "priority_tier",
    "business_risk_tier"
]

st.dataframe(
    top_exposure[exposure_columns],
    width="stretch",
    hide_index=True
)

# ============================================================
# INTERVENTION SIMULATOR
# ============================================================

st.divider()

st.subheader("🎯 Intervention Simulator")

st.write(
    "Estimate how much revenue risk can be captured by targeting "
    "the highest-risk orders."
)
# -----------------------------
# User controls
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    intervention_level = st.selectbox(
        "Orders to Target",
        ["Top 1%", "Top 2%", "Top 5%", "Top 10%", "Top 20%"],
        index=2
    )

with col2:
    prevention_effectiveness_pct = st.slider(
        "Prevention Effectiveness",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        format="%d%%"
    )

with col3:
    cost_per_order = st.selectbox(
        "Intervention Cost per Order",
        [10, 25, 50, 75, 100],
        index=0,
        format_func=lambda x: f"₹{x}"
    )

prevention_effectiveness = prevention_effectiveness_pct / 100

# -----------------------------
# Convert intervention level
# -----------------------------

intervention_share = {
    "Top 1%": 0.01,
    "Top 2%": 0.02,
    "Top 5%": 0.05,
    "Top 10%": 0.10,
    "Top 20%": 0.20
}

share = intervention_share[intervention_level]

orders_targeted = int(len(filtered_df) * share)

# -----------------------------
# Revenue at risk captured
# -----------------------------

targeted_orders = (
    filtered_df
    .sort_values("revenue_at_risk", ascending=False)
    .head(orders_targeted)
)

targeted_revenue_risk = targeted_orders["revenue_at_risk"].sum()

potential_savings = (
    targeted_revenue_risk * prevention_effectiveness
)

# -----------------------------
# Financial impact
# -----------------------------

intervention_cost = (
    orders_targeted * cost_per_order
)

net_benefit = (
    potential_savings - intervention_cost
)

roi = (
    net_benefit / intervention_cost
    if intervention_cost > 0
    else 0
)

# -----------------------------
# Display financial results
# -----------------------------

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:
    st.metric(
        "Orders Targeted",
        f"{orders_targeted:,}"
    )

with metric2:
    st.metric(
        "Revenue Risk Targeted",
        f"₹{targeted_revenue_risk:,.0f}"
    )

with metric3:
    st.metric(
        "Potential Savings",
        f"₹{potential_savings:,.0f}"
    )

with metric4:
    st.metric(
        "Intervention Cost",
        f"₹{intervention_cost:,.0f}"
    )

with metric5:
    st.metric(
        "Net Benefit",
        f"₹{net_benefit:,.0f}"
    )

st.metric(
    "Estimated ROI",
    f"{roi:.1%}"
)

st.markdown("### 💡 Business Recommendation")

if net_benefit > 0:

    st.success(
        f"Targeting {intervention_level.lower()} of orders is financially viable. "
        f"At {prevention_effectiveness_pct}% prevention effectiveness and "
        f"₹{cost_per_order} intervention cost per order, the estimated "
        f"net benefit is ₹{net_benefit:,.0f} with an ROI of {roi:.1%}."
    )

else:

    st.error(
        f"This intervention strategy is not financially viable under the "
        f"current assumptions. The estimated net loss is "
        f"₹{abs(net_benefit):,.0f}."
    )

# ============================================================
# OPTIMAL INTERVENTION STRATEGY
# ============================================================

st.markdown("---")

st.header("🎯 Optimal Intervention Strategy")

st.write(
    "Compare different intervention levels to determine how much "
    "revenue risk can be captured by targeting high-risk orders."
)

# Intervention levels to evaluate
intervention_levels = {
    "Top 1%": 0.01,
    "Top 2%": 0.02,
    "Top 5%": 0.05,
    "Top 10%": 0.10,
    "Top 20%": 0.20
}

strategy_rows = []

for level, share in intervention_levels.items():
   
    orders_targeted_strategy = int(len(filtered_df) * share)

    targeted_orders_strategy = (
    filtered_df
    .sort_values("revenue_at_risk", ascending=False)
    .head(orders_targeted_strategy)
    )

    revenue_risk_strategy = (
        targeted_orders_strategy["revenue_at_risk"].sum()
    )

    savings_strategy = (
        revenue_risk_strategy * prevention_effectiveness
    )

    intervention_cost_strategy = (
        orders_targeted_strategy * cost_per_order
    )

    net_benefit_strategy = (
        savings_strategy - intervention_cost_strategy
    )

    roi_strategy = (
        net_benefit_strategy / intervention_cost_strategy
        if intervention_cost_strategy > 0
        else 0
    )

    strategy_rows.append({
        "Intervention Level": level,
        "Orders Targeted": orders_targeted_strategy,
        "Revenue Risk Captured": revenue_risk_strategy,
        "Potential Savings": savings_strategy,
        "Intervention Cost": intervention_cost_strategy,
        "Net Benefit": net_benefit_strategy,
        "ROI": roi_strategy
    })

strategy_df = pd.DataFrame(strategy_rows)

# Display comparison table

st.dataframe(
    strategy_df.style.format({
        "Orders Targeted": "{:,.0f}",
        "Revenue Risk Captured": "₹{:,.0f}",
        "Potential Savings": "₹{:,.0f}",
        "Intervention Cost": "₹{:,.0f}",
        "Net Benefit": "₹{:,.0f}",
        "ROI": "{:.1%}"
    }),
    width="stretch",
    hide_index=True
)

# ============================================================
# BEST INTERVENTION STRATEGY
# ============================================================

st.markdown("### 🏆 Recommended Intervention Strategy")

# Find strategy with the highest net benefit
best_strategy = strategy_df.loc[
    strategy_df["Net Benefit"].idxmax()
]

best_level = best_strategy["Intervention Level"]
best_orders = int(best_strategy["Orders Targeted"])
best_revenue_risk = best_strategy["Revenue Risk Captured"]
best_savings = best_strategy["Potential Savings"]
best_cost = best_strategy["Intervention Cost"]
best_net_benefit = best_strategy["Net Benefit"]
best_roi = best_strategy["ROI"]

if best_net_benefit > 0:

    st.success(
        f"""
        **Recommended Strategy: {best_level}**

        Target approximately **{best_orders:,} high-risk orders**.

        • Revenue risk targeted: **₹{best_revenue_risk:,.0f}**  
        • Potential savings: **₹{best_savings:,.0f}**  
        • Intervention cost: **₹{best_cost:,.0f}**  
        • Expected net benefit: **₹{best_net_benefit:,.0f}**  
        • Estimated ROI: **{best_roi:.1%}**

        This strategy provides the highest expected financial benefit
        among the evaluated intervention levels under the current
        assumptions.
        """
    )

else:

    st.warning(
        f"""
        None of the evaluated intervention strategies produces a
        positive net benefit under the current assumptions.

        The strategy with the highest financial outcome is
        **{best_level}**, but its estimated net benefit is
        **₹{best_net_benefit:,.0f}**.

        The business should reduce intervention cost or improve
        prevention effectiveness before implementing the strategy.
        """
    )

# ============================================================
# NET BENEFIT COMPARISON
# ============================================================

st.markdown("### 📊 Net Benefit by Intervention Level")

fig_strategy = px.bar(
    strategy_df,
    x="Intervention Level",
    y="Net Benefit",
    text="Net Benefit",
    title="Expected Net Benefit by Intervention Level"
)

fig_strategy.update_traces(
    texttemplate="₹%{text:,.0f}",
    textposition="outside"
)

fig_strategy.update_layout(
    xaxis_title="Intervention Level",
    yaxis_title="Net Benefit (₹)",
    showlegend=False
)

st.plotly_chart(
    fig_strategy,
    width="stretch"
)

st.markdown("---")

st.header("🚨 Top High-Risk Orders")

st.write(
    "These orders have the highest predicted return risk and financial exposure."
)

top_risk_orders = filtered_df.sort_values(         "revenue_at_risk",
    ascending=False
).head(10)

top_risk_columns = [
    "order_id",
    "product_category",
    "customer_segment",
    "risk_prob",
    "net_order_value",
    "revenue_at_risk",
    "expected_return_loss",
    "priority_tier",
    "business_risk_tier"
]

st.dataframe(
    top_risk_orders[top_risk_columns].round(3),
    width="stretch",
    hide_index=True
)

st.markdown("---")

st.header("💡 Recommended Business Actions")

st.markdown("""
### 1️⃣ Critical Risk Orders
Contact the customer before fulfillment or delivery confirmation.
Consider additional product information, confirmation calls, or proactive support.

### 2️⃣ High Risk Orders
Monitor these orders closely and provide clearer product information,
delivery expectations, and return-policy communication.

### 3️⃣ High-Value + High-Risk Orders
Prioritize these orders because they create the greatest potential
financial exposure.

### 4️⃣ Low-Risk Orders
No special intervention is required. Continue the normal order process.
""")

# ============================================================
# BUSINESS RISK INSIGHTS
# ============================================================
st.markdown("---")

st.header("📊 Business Risk Insights")

# Risk tier summary
risk_summary = filtered_df.groupby("business_risk_tier").agg(
    orders=("order_id", "count"),
    revenue_at_risk=("revenue_at_risk", "sum"),
    expected_return_loss=("expected_return_loss", "sum")
).reset_index()

risk_summary["revenue_at_risk"] = (
    risk_summary["revenue_at_risk"].round(2)
)

risk_summary["expected_return_loss"] = (
    risk_summary["expected_return_loss"].round(2)
)

st.dataframe(
    risk_summary,
    width="stretch",
    hide_index=True
)

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("Risk Distribution")

fig_risk_distribution = px.bar(
    risk_summary,
    x="business_risk_tier",
    y="orders",
    text="orders",
    title="Number of Orders by Business Risk Tier"
)

fig_risk_distribution.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)

fig_risk_distribution.update_layout(
    xaxis_title="Business Risk Tier",
    yaxis_title="Number of Orders",
    showlegend=False
)

st.plotly_chart(
    fig_risk_distribution,
    width="stretch"
)

# ============================================================
# RISK VS REVENUE
# ============================================================

st.subheader("Risk Probability vs Order Value")

fig_risk_value = px.scatter(
    filtered_df,
    x="risk_prob",
    y="net_order_value",
    color="business_risk_tier",
    hover_data=[
        "order_id",
        "revenue_at_risk",
        "priority_tier"
    ],
    title="Return Risk vs Order Value"
)

fig_risk_value.update_layout(
    xaxis_title="Return Risk Probability",
    yaxis_title="Net Order Value"
)

st.plotly_chart(
    fig_risk_value,
    width="stretch"
)

# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

st.subheader("Customer Segment Risk")

segment_summary = (
    filtered_df
    .groupby("customer_segment")
    .agg(
        orders=("order_id", "count"),
        revenue_at_risk=("revenue_at_risk", "sum"),
        avg_risk=("risk_prob", "mean")
    )
    .reset_index()
)

segment_summary["avg_risk"] = (
    segment_summary["avg_risk"] * 100
).round(2)

segment_summary["revenue_at_risk"] = (
    segment_summary["revenue_at_risk"].round(2)
)

st.dataframe(
    segment_summary,
    use_container_width=True,
    hide_index=True
)

