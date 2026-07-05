🚀 [Live Dashboard](https://ecommerce-pyspark-analysis-ckdjq8q6olrgtzf7a9kxfo.streamlit.app/)

# E-commerce Customer Segmentation & Revenue Analysis

Identified which customers actually drive revenue in a 541,909-transaction
e-commerce dataset — RFM segmentation surfaced ~1,300 "Champion" customers
generating nearly £7M in revenue, out of hundreds of thousands of transactions,
using a PySpark pipeline built on Spark SQL.

## The Problem

Most businesses treat all customers the same when deciding where to spend
retention and marketing budget. That's expensive and inefficient — a small
fraction of customers usually drive most of the revenue, and without
segmentation, there's no way to know who they are or prioritize accordingly.

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
- **Visualizations** — monthly revenue trend, RFM segment breakdown, recency
  vs. monetary scatter

## Key Findings

- ~1,300 Champion customers generated nearly £7M in revenue — a small
  fraction of the customer base driving the majority of revenue
- November 2011 peaked at £1.16M in monthly revenue
- September 2011 saw +47.6% month-over-month growth, consistent with a
  holiday-season demand surge
- Netherlands had the highest revenue-per-customer ratio outside the UK,
  suggesting an underexplored high-value market

## Why This Matters (Business Impact)

Without this segmentation, a business would spend equally on retaining a
£10 one-time buyer and a £5,000/year Champion. RFM segmentation turns "all
customers" into "these specific ~1,300 customers, protect them first" — a
directly actionable prioritization, not just a descriptive chart.

## Limitations

- This is a segmentation and analysis project, not a predictive model —
  it tells you *who* matters today, not who is likely to churn tomorrow.
- RFM scoring uses fixed thresholds; it doesn't adapt automatically to
  shifts in overall customer behavior over time.
- The dataset spans Dec 2010–Dec 2011 only; segment cutoffs (what counts as
  "recent" or "high monetary value") would need re-tuning on fresh data.

## Future Work

- Add a churn prediction model on top of these RFM segments, so at-risk
  Champions get flagged before they lapse, not just identified after the fact
- Add a decision layer that translates a churn flag into a specific action
  (discount, outreach, or no action if the customer isn't worth the retention
  spend) — moving from "here's the data" to "here's what to do about it"
- Automate segment refresh on a recurring schedule (currently a one-time
  batch run) so segments stay current as new transactions come in

## Tech Stack

- PySpark 4.1 | Java 17
- Jupyter Notebook
- Matplotlib | Seaborn
- Streamlit (dashboard)

## Dataset

UCI Online Retail Dataset — 541,909 transactions, Dec 2010 – Dec 2011

## How to Run

**Fastest way — no setup:** open the [live dashboard](https://ecommerce-pyspark-analysis-ckdjq8q6olrgtzf7a9kxfo.streamlit.app/) directly. It runs the full segmentation and analysis interactively — no install required.

**To explore the underlying pipeline:**
1. Clone this repo
2. Install PySpark 4.1 and Java 17
3. Open the Jupyter notebook and run cells top to bottom — the UCI Online Retail dataset is [publicly available here](https://archive.ics.uci.edu/dataset/352/online+retail) if not already included in the repo
