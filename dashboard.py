import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")
    df.columns = df.columns.str.strip().str.replace("ï»¿", "", regex=False)
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

# ── Churn model: trained with a TIME-BASED split (no label leakage) ──────────
# Features come from behavior BEFORE the cutoff; the label comes from whether
# the customer purchased again AFTER the cutoff. This is what makes the model
# predict something genuinely unknown, instead of re-deriving the RFM segment
# it was given as input.
@st.cache_resource
def train_churn_model(_df):
    last_date = _df["InvoiceDate"].max()
    cutoff_date = last_date - pd.Timedelta(days=90)

    pre_cutoff = _df[_df["InvoiceDate"] < cutoff_date]
    post_cutoff = _df[_df["InvoiceDate"] >= cutoff_date]

    features = (pre_cutoff
        .groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalRevenue", "sum")
        )
        .reset_index()
    )
    features["Monetary"] = features["Monetary"].round(2)

    active_after_cutoff = set(post_cutoff["CustomerID"].unique())
    features["Churn"] = features["CustomerID"].apply(
        lambda cid: 0 if cid in active_after_cutoff else 1
    )

    X = features[["Recency", "Frequency", "Monetary"]]
    y = features["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Active", "Churned"], output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    # Score EVERY current customer using the full dataset up to today, so the
    # dashboard's "who's at risk right now" view is a real forward-looking
    # prediction, not a restated segment label.
    current_features = (_df
        .groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (last_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalRevenue", "sum")
        )
        .reset_index()
    )
    current_features["Monetary"] = current_features["Monetary"].round(2)
    current_features["Churn_Probability"] = model.predict_proba(
        current_features[["Recency", "Frequency", "Monetary"]]
    )[:, 1]

    return model, report, auc, current_features

model, report, auc, current_features = train_churn_model(df)

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

# ── Churn Prediction section ──────────────────────────────────────────────────
st.subheader("🤖 Churn Prediction — Random Forest")
st.caption(
    "Predicts whether a customer will make **zero purchases in the next 90 days**, "
    "using only Recency/Frequency/Monetary as they stood *before* that window. "
    "Trained and evaluated on separate time periods so the model can't see the "
    "answer in its own inputs."
)

perf_col, predict_col = st.columns(2)

with perf_col:
    st.markdown("**Model Performance**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{report['accuracy']*100:.0f}%")
    m2.metric("ROC-AUC Score", f"{auc:.4f}")
    m3.metric("Churned Recall", f"{report['Churned']['recall']*100:.0f}%")

    st.markdown("**Feature Importance**")
    feat_importance = pd.DataFrame({
        "Feature": ["Recency", "Frequency", "Monetary"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.barh(feat_importance["Feature"], feat_importance["Importance"], color="#4C72B0")
    ax4.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig4)

with predict_col:
    st.markdown("**🔮 Predict Churn for a Customer**")
    recency_input = st.slider("Recency (days since last purchase)", 0, 400, 50)
    frequency_input = st.slider("Frequency (number of orders)", 1, 250, 5)
    monetary_input = st.slider("Monetary (total spend £)", 0, 10000, 1000)

    input_df = pd.DataFrame({
        "Recency": [recency_input],
        "Frequency": [frequency_input],
        "Monetary": [monetary_input]
    })
    risk = model.predict_proba(input_df)[0, 1]

    if risk < 0.33:
        st.success(f"✅ Low Churn Risk: {risk:.1%}")
    elif risk < 0.66:
        st.warning(f"⚠️ Medium Churn Risk: {risk:.1%}")
    else:
        st.error(f"🚨 High Churn Risk: {risk:.1%}")

    st.markdown("**Top 10 High Risk Customers**")
    high_risk = (current_features
        .sort_values("Churn_Probability", ascending=False)
        .head(10)
        [["CustomerID", "Recency", "Frequency", "Monetary", "Churn_Probability"]]
        .reset_index(drop=True))
    high_risk["Monetary"] = high_risk["Monetary"].apply(lambda x: f"£{x:,.2f}")
    high_risk["Churn_Probability"] = high_risk["Churn_Probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(high_risk, use_container_width=True)

st.markdown("---")
st.caption("Built with PySpark · Airflow · Streamlit | Abdullah Bootwala")
