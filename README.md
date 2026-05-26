# E-commerce Sales Analysis with PySpark

Big Data pipeline built on 541,909 real e-commerce transactions using PySpark, RFM customer segmentation, and Spark SQL.

## Tech Stack
- PySpark 4.1 | Java 17
- Jupyter Notebook
- Matplotlib | Seaborn

## What's inside
- **Data cleaning** — removed nulls, cancellations, bad entries (397,884 clean rows)
- **Feature engineering** — TotalRevenue, Month, Year columns
- **RFM Segmentation** — Champions, Loyal, At Risk, Lost
- **Spark SQL** — GROUP BY, HAVING, subqueries, RANK, LAG, JOINs
- **Visualisations** — monthly revenue trend, RFM segments, recency vs monetary scatter

## Key Findings
- November 2011 peaked at £1.16M revenue
- September 2011: +47.6% MoM growth — holiday season surge
- ~1,300 Champion customers generated nearly £7M in revenue
- Netherlands had highest revenue-per-customer ratio outside UK

## Dataset
UCI Online Retail Dataset — 541,909 transactions, Dec 2010 – Dec 2011