🚀 [Live Dashboard](https://ecommerce-pyspark-analysis-ckdjq8q6olrgtzf7a9kxfo.streamlit.app/)

# E-commerce Customer Segmentation & Churn Prediction

Identified which customers actually drive revenue in a 541,909-transaction
e-commerce dataset — RFM segmentation surfaced ~1,300 "Champion" customers
generating nearly £7M in revenue — then built a churn model to predict which
customers are likely to go quiet in the next 90 days, using a PySpark and
Spark SQL pipeline with an automated Airflow refresh.

## The Problem

Most businesses treat all customers the same when deciding where to spend
retention and marketing budget. That's expensive and inefficient — a small
fraction of customers usually drive most of the revenue, and without
segmentation and forward-looking risk scoring, there's no way to know who
matters most or who's about to leave.

## Approach

- **Data cleaning** — removed nulls, cancellations, and bad entries, taking
  541,909 raw transactions down to 397,884 clean rows (~26% of the raw data
  was noise — cancellations, missing customer IDs, negative quantities)
- **Feature engineering** — derived TotalRevenue, Month, and Year fields
  needed for time-based and segment-based analysis
- **RFM Segmentation** — scored every customer on Recency, Frequency, and
  Monetary value, then grouped into Champions, Loyal, At Risk, and Lost
  segments
- **Spark SQL analysis** — GROUP BY, HAVING, subqueries, RANK, LAG, and JOINs
  to answer specific business questions (which months grew, which segments
  matter, which geographies overperform)
- **Churn prediction** — a Random Forest model trained with a time-based
  split (see below) to predict which customers are likely to make zero
  purchases in the next 90 days
- **Visualizations** — monthly revenue trend, RFM segment breakdown, recency
  vs. monetary scatter, feature importance, confusion matrix

## Key Findings

- ~1,300 Champion customers generated nearly £7M in revenue — a small
  fraction of the customer base driving the majority of revenue
- November 2011 peaked at £1.16M in monthly revenue
- September 2011 saw +47.6% month-over-month growth, consistent with a
  holiday-season demand surge
- Netherlands had the highest revenue-per-customer ratio outside the UK,
  suggesting an underexplored high-value market
- **Monetary value, not recency, is the strongest churn predictor** in this
  dataset (see Feature Importance below) — customers who've spent less
  overall are meaningfully more likely to go quiet, independent of how
  recently they last purchased

## Churn Prediction — and the Bug I Caught Along the Way

**Model:** Random Forest Classifier | **Features:** Recency, Frequency,
Monetary | **Target:** made zero purchases in the 90 days following a
cutoff date

**Results (after the fix below):** 62% accuracy, 0.67 ROC-AUC, 57% recall
on churned customers.

### The leakage bug

My first version of this model reported 96% accuracy and a 0.9955 ROC-AUC —
suspiciously close to perfect. Digging in, I found the churn label had been
derived directly from the RFM segment (`Churn = 1 if Segment in ["At Risk",
"Lost"]`), and the RFM segment itself was computed from the same Recency,
Frequency, and Monetary values used as the model's only features. The model
wasn't predicting anything — it was just re-deriving a labeling rule I'd
handed it, dressed up as a churn probability that snapped to 0% or 100% for
every customer instead of showing a real gradient.

**The fix:** a genuine time-based train/test split. Features are computed
from customer behavior *before* a 90-day cutoff; the label (churned or not)
is determined by whether the customer purchased again *after* that cutoff.
Since the label now comes from a period the model has never seen at
feature-computation time, there's no leakage — and the resulting 0.67
ROC-AUC, while lower, is an honest signal instead of an inflated one.

This is the version currently live on the dashboard.

## Why This Matters (Business Impact)

Without RFM segmentation, a business would spend equally on retaining a £10
one-time buyer and a £5,000/year Champion. Without a genuinely predictive
churn model, "risk scores" are just relabeled historical categories, not an
early warning system. Together, this project turns "all customers" into
"these ~1,300 Champions, protect them first" and "these specific accounts,
flagged before they go quiet" — directly actionable prioritization, not just
descriptive charts.

## Limitations

- RFM scoring uses fixed thresholds; it doesn't adapt automatically to
  shifts in overall customer behavior over time.
- The dataset spans Dec 2010–Dec 2011 only; segment cutoffs and the churn
  model's 90-day window would need re-tuning on fresh, ongoing data.
- The churn model only uses Recency, Frequency, and Monetary — no product
  category, seasonality, or marketing-touch features — so 0.67 ROC-AUC
  likely represents close to the ceiling of what these three features alone
  can predict. Adding richer features is the most direct way to improve it.
- The live dashboard's country filter currently only applies to the revenue
  trend chart, not to RFM segments or churn scores, since those are
  precomputed globally rather than per-country.

## Future Work

- Add a decision layer that translates a churn score into a specific action
  (discount, outreach, or no action if the customer isn't worth the
  retention spend) — moving from "here's the risk" to "here's what to do
  about it"
- Expand churn features beyond RFM (e.g., product category diversity,
  average order value trend) to push past the current feature ceiling
- Automate segment and churn-score refresh on a recurring schedule via the
  existing Airflow DAG (currently a manual batch run)
- Extend country filtering to RFM and churn sections by recomputing
  segmentation per-country

## Tech Stack

- PySpark 4.1 | Java 17
- Jupyter Notebook
- Scikit-learn (Random Forest)
- Matplotlib | Seaborn
- Streamlit (dashboard)
- Airflow (pipeline scheduling)

## Dataset

UCI Online Retail Dataset — 541,909 transactions, Dec 2010 – Dec 2011

## How to Run

**Fastest way — no setup:** open the [live dashboard](https://ecommerce-pyspark-analysis-ckdjq8q6olrgtzf7a9kxfo.streamlit.app/) directly. It runs the full segmentation, revenue analysis, and churn prediction interactively — no install required.

**To explore the underlying pipeline:**
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt` (includes PySpark 4.1, Java 17, scikit-learn, Streamlit)
3. Open the notebooks in `notebooks/` and run cells top to bottom — the UCI Online Retail dataset is [publicly available here](https://archive.ics.uci.edu/dataset/352/online+retail) if not already included in `data/`
4. Run the dashboard locally with `streamlit run dashboard.py`