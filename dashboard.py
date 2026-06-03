import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/online_retail.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # remove extra spaces from column names
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.month
    df["Year"] = df["InvoiceDate"].dt.year
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df

@st.cache_data
def load_rfm():
    rfm = pd.read_csv("output/rfm_segments.csv")
    return rfm

df = load_data()
rfm = load_rfm()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 E-commerce Sales Dashboard")
st.markdown("**Dataset:** UCI Online Retail | 541,909 transactions | Dec 2010 – Dec 2011")
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{df['TotalRevenue'].sum():,.0f}")
col2.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
col3.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
col4.metric("Countries", f"{df['Country'].nunique()}")

st.markdown("---")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
countries = ["All"] + sorted(df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

if selected_country != "All":
    df_filtered = df[df["Country"] == selected_country]
else:
    df_filtered = df

# ── Monthly revenue trend ─────────────────────────────────────────────────────
st.subheader("📈 Monthly Revenue Trend")
monthly = df_filtered.groupby("YearMonth")["TotalRevenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["YearMonth"], monthly["TotalRevenue"], marker="o", color="#4C72B0", linewidth=2.5)
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (£)")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── Two column layout ─────────────────────────────────────────────────────────
left, right = st.columns(2)

# Top 10 countries
with left:
    st.subheader("🌍 Top 10 Countries by Revenue")
    top_countries = (df.groupby("Country")["TotalRevenue"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(10)
                     .reset_index())
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.barh(top_countries["Country"][::-1], top_countries["TotalRevenue"][::-1], color="#4C72B0")
    ax2.set_xlabel("Revenue (£)")
    plt.tight_layout()
    st.pyplot(fig2)

# RFM segments
with right:
    st.subheader("👥 RFM Customer Segments")
    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#95a5a6"]
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.barh(seg_counts["Segment"], seg_counts["Customers"], color=colors)
    ax3.set_xlabel("Number of Customers")
    plt.tight_layout()
    st.pyplot(fig3)

st.markdown("---")

# ── RFM table ─────────────────────────────────────────────────────────────────
st.subheader("🏆 Top Champion Customers")
champions = (rfm[rfm["Segment"] == "Champions"]
             .sort_values("Monetary", ascending=False)
             .head(10)
             [["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Score", "Segment"]]
             .reset_index(drop=True))
champions["Monetary"] = champions["Monetary"].apply(lambda x: f"£{x:,.2f}")
st.dataframe(champions, use_container_width=True)

st.markdown("---")
st.caption("Built with PySpark · Airflow · Streamlit | Abdullah Bootwala")