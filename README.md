# Return or Revenue?
### Finding the Hidden Cost of E-Commerce Returns Using Machine Learning & Business Analytics

**A Data Analytics | Machine Learning | Business Intelligence Project**

> Don't just predict what customers buy. Identify which orders could put your revenue at risk.

---

## 1. Project Overview

E-commerce returns are more than just refunds. Every returned order can create additional costs through reverse logistics, inspection, repackaging, inventory handling, customer support and potential product depreciation.

But what if businesses could identify risky orders before they become expensive?

**Return or Revenue?** is a Machine Learning and Business Analytics project that predicts the probability of an e-commerce order being returned and translates those predictions into financial exposure and actionable business insights.

The project combines predictive modelling, financial risk analysis and an interactive Streamlit dashboard to help businesses understand where their return-related risks are concentrated and how they can prioritise preventive actions.

## 2. Business Problem

E-commerce businesses often focus on sales revenue without fully understanding the financial impact of product returns.

The major challenges include:

- Identifying orders with a high probability of being returned.
- Understanding the financial exposure associated with risky orders.
- Finding product categories and customer segments contributing to return risk.
- Prioritising limited resources towards orders that deserve attention.
- Evaluating whether preventive interventions could provide financial benefits.

### Proposed Solution

This project develops a predictive and decision-support system that:

1. Analyses historical e-commerce order data.
2. Predicts the probability of each order being returned.
3. Estimates revenue exposure using return probabilities and order values.
4. Classifies orders into business risk and priority categories.
5. Analyses risk across products and customer segments.
6. Simulates different intervention strategies to estimate potential savings and ROI.

**Core Concept:**

`Return Probability × Order Value = Revenue at Risk`

## 3. Project Objectives

- Develop a machine learning model to predict e-commerce product returns.
- Compare multiple classification algorithms.
- Identify important patterns in customer and product return behaviour.
- Estimate financial exposure associated with predicted returns.
- Develop an interactive business risk dashboard.
- Simulate intervention strategies and evaluate their estimated financial benefits.
- Convert machine learning predictions into meaningful business recommendations.

## 4. Key Dashboard Metrics

The dashboard analyses **60,000 e-commerce orders**.

| Business Metric | Project Result |
|---|---:|
| Total Orders | 60,000 |
| Overall Return Rate | 47.5% |
| Revenue at Risk | ₹7,17,910 |
| Expected Return Loss | ₹2,50,266 |
| High + Critical Business Risk Orders | 78 |

*These are estimates derived from the project dataset, not verified real-world company financial losses.*

## 5. Machine Learning Approach

The project compares multiple classification algorithms to identify a suitable model for predicting product returns.

### Models Compared

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.5894 |
| Random Forest | 0.5913 |
| Gradient Boosting | **0.5955** |
| Decision Tree | 0.5701 |
| Naive Bayes | 0.5697 |
| Dummy Classifier | 0.5000 |

### Selected Model: Gradient Boosting

Gradient Boosting achieved the highest validation ROC-AUC score among the compared models.

The selected implementation uses `HistGradientBoostingClassifier`.

### Holdout Performance

| Evaluation Metric | Score |
|---|---:|
| ROC-AUC | 0.5928 |
| Precision | 0.5623 |
| Recall | 0.4342 |
| F1-Score | 0.4900 |

The model's predictive performance is modest. Therefore, it is presented as a risk-ranking and decision-support mechanism rather than a perfect return prediction system.

### Model Development Workflow

1. Data loading and initial inspection.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature preparation and transformation.
5. Training multiple classification algorithms.
6. Comparing model performance using validation metrics.
7. Selecting the best-performing model.
8. Evaluating the selected model on holdout data.
9. Generating return probabilities using `predict_proba()`.

## 6. Financial Risk & Business Analysis

A major objective of this project is to convert machine learning predictions into business-oriented information.

### Revenue at Risk

Revenue at Risk is estimated using:

`Revenue at Risk = Return Probability × Net Order Value`

**Example:**

- Net Order Value = ₹1,000
- Predicted Return Probability = 70%

Estimated Revenue at Risk = ₹700.

This represents financial exposure, not necessarily an actual financial loss.

### Expected Return Loss

The dashboard also analyses expected return-related loss using the estimated financial cost associated with returns and their predicted probabilities.

### Business Risk Classification

Orders are classified into four categories:

- Low
- Medium
- High
- Critical

A separate priority classification helps identify orders that deserve business attention.

## 7. Interactive Intervention Simulator

One of the key features of this project is an interactive business intervention simulator.

It allows users to experiment with:

- Percentage of orders to target.
- Assumed prevention effectiveness.
- Intervention cost per order.

### Financial Calculations

**Potential Savings**

`Targeted Revenue at Risk × Prevention Effectiveness`

**Intervention Cost**

`Targeted Orders × Cost per Order`

**Net Benefit**

`Potential Savings − Intervention Cost`

**Return on Investment (ROI)**

`(Net Benefit / Intervention Cost) × 100`

### Intervention Strategy Comparison

| Target Orders | Net Benefit | Estimated ROI |
|---|---:|---:|
| Top 1% | ₹9,566 | 159.4% |
| Top 2% | ₹14,593 | 121.6% |
| Top 5% | ₹22,199 | 74.0% |
| Top 10% | **₹22,585** | 37.6% |
| Top 20% | ₹2,830 | 2.4% |

The Top 10% strategy produces the highest estimated net benefit in this particular simulation, while the Top 1% strategy has the highest ROI percentage.

*Note: Prevention effectiveness is a user-defined business assumption, not a value predicted by the ML model. The simulated savings are hypothetical and should be validated through real-world pilot experiments.*

## 8. Key Business Insights

The dashboard helps uncover several interesting patterns:

- **Toys:** Highest aggregate revenue at risk among the analysed product categories.
- **Clothing:** Highest observed return rate among product categories.
- **Regular Customers:** Highest total revenue exposure because of their large order volume.
- **New / Unproven Customers:** Highest observed return rate among customer segments.
- **Low Business Risk Orders:** Represent the largest total exposure by volume, despite having lower individual risk classifications.

These findings demonstrate why businesses should consider both individual risk and aggregate financial exposure.

## 9. Dashboard Features

The Streamlit dashboard provides:

- Executive-level business risk overview.
- Total orders, return rate and financial exposure metrics.
- Product category risk analysis.
- Customer segment risk analysis.
- Business risk distribution.
- Priority tier distribution.
- High-priority order identification.
- Financial exposure analysis.
- Return probability versus order value analysis.
- Interactive intervention simulator.
- ROI and net benefit comparison.
- Recommended business actions.

## 10. Technology Stack

| Category | Tools & Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Predictive Modelling | Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, Naive Bayes |
| Dashboard Development | Streamlit |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git & GitHub |

## 11. Project Structure

```text
return-or-revenue/
│
├── dashboard/
├── data/
├── docs/
├── models/
│
├── notebooks/
│   ├── 01_load_and_check.ipynb
│   ├── 02_eda_and_kpis.ipynb
│   ├── 03_models.ipynb
│   ├── 04_holdout_and_matrix.ipynb
│   ├── 06_business_impact.ipynb
│   ├── app.py
│   └── final_business_risk_dataset.csv
│
├── reports/
├── src/
├── config.py
├── README.md
└── requirements.txt
```

## 12. Installation & Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/KASHISH823/return-or-revenue.git
```

### Step 2: Navigate to the Project Folder

```bash
cd return-or-revenue
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Dashboard

```bash
streamlit run notebooks/app.py
```

The dashboard should open automatically in your web browser.

*Note: The dashboard requires its CSV dataset to be available at the file path configured in `app.py`.*

## 13. Future Improvements

- Improve predictive performance through feature engineering and hyperparameter tuning.
- Validate predictions using additional datasets.
- Incorporate more detailed reverse-logistics and product recovery costs.
- Test intervention effectiveness through controlled business experiments.
- Introduce model monitoring and prediction drift detection.
- Develop more advanced customer behaviour analysis.

## 14. Learning Outcomes

Working on this project provided practical experience in:

- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Machine learning model development and comparison.
- Classification model evaluation.
- Financial exposure estimation.
- Customer and product behaviour analysis.
- Interactive dashboard development.
- Business decision-making through predictive analytics.
- Communicating analytical findings through business insights.

## 15. Live Project

**Deployment Platform:** Streamlit Community Cloud

**Live Dashboard:** Coming Soon

Once deployment is complete, the live application link will be added here.

---

## About the Author

**Kashish Kumari**

BCA Student | Aspiring Data Analyst | Machine Learning & Business Analytics Enthusiast

**GitHub:** [KASHISH823](https://github.com/KASHISH823)

---

<p align="center">
  <b>Turning Data into Decisions. Turning Predictions into Business Insights.</b>
</p>

<p align="center">
  If you find this project interesting, consider giving the repository a star!
</p>
