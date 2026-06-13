import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Build Everything from CSV ────────────────────────────────
@st.cache_data
def load_data():
    # Step 1 - Load raw CSV
    df = pd.read_csv("data.csv", encoding="latin-1")

    # Step 2 - Clean data
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["CustomerID"].notna()]
    df = df[df["Quantity"]   > 0]
    df = df[df["UnitPrice"]  > 0]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Revenue"]    = df["Quantity"] * df["UnitPrice"]

    # Step 3 - RFM calculation
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum")
    ).reset_index()

    # Step 4 - Log transform
    rfm["R_log"] = np.log1p(rfm["Recency"])
    rfm["F_log"] = np.log1p(rfm["Frequency"])
    rfm["M_log"] = np.log1p(rfm["Monetary"])

    # Step 5 - Scale
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["R_log", "F_log", "M_log"]])

    # Step 6 - KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["KMeans_Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Step 7 - Label segments
    profile = rfm.groupby("KMeans_Cluster")[
        ["Recency","Frequency","Monetary"]
    ].mean()

    segment_map = {
        profile["Monetary"].idxmax()  : "🏆 Champions",
        profile["Recency"].idxmin()   : "🌱 New/Recent Customers",
        profile["Frequency"].idxmin() : "💤 Lost Customers",
        profile["Recency"].idxmax()   : "⚠️ At Risk"
    }
    rfm["Segment"] = rfm["KMeans_Cluster"].map(segment_map)

    return rfm, scaler, kmeans

# ── Load Data ────────────────────────────────────────────────
rfm, scaler, kmeans = load_data()

# Segment color mapping
COLORS = {
    "🏆 Champions"            : "#FFD700",
    "🌱 New/Recent Customers" : "#90EE90",
    "⚠️ At Risk"              : "#FF6347",
    "💤 Lost Customers"       : "#A9A9A9"
}

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🛒 Segmentation App")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "👥 Segments", "🔍 Customer Lookup", "🤖 Predict Segment"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Customers:** {len(rfm):,}")
st.sidebar.markdown(f"**Segments:** {rfm['Segment'].nunique()}")

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.title("📊 Customer Segmentation — Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",  f"{len(rfm):,}")
    col2.metric("Avg Recency",      f"{rfm['Recency'].mean():.0f} days")
    col3.metric("Avg Frequency",    f"{rfm['Frequency'].mean():.1f} orders")
    col4.metric("Avg Monetary",     f"£{rfm['Monetary'].mean():.2f}")

    st.markdown("---")

    st.subheader("📈 RFM Distributions")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.histogram(rfm, x="Recency", nbins=50,
                           title="Recency Distribution",
                           color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(rfm, x="Frequency", nbins=50,
                           title="Frequency Distribution",
                           color_discrete_sequence=["darkorange"])
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.histogram(rfm, x="Monetary", nbins=50,
                           title="Monetary Distribution",
                           color_discrete_sequence=["green"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🌐 Customers in 3D RFM Space")
    fig = px.scatter_3d(
        rfm,
        x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        opacity=0.6,
        title="Customer RFM Space",
        color_discrete_map=COLORS
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTS
# ════════════════════════════════════════════════════════════
elif page == "👥 Segments":

    st.title("👥 Customer Segments")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            seg_counts,
            values="Count",
            names="Segment",
            title="Segment Distribution",
            color="Segment",
            color_discrete_map=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            title="Customer Count per Segment",
            color="Segment",
            color_discrete_map=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📋 Segment Profiles (Mean RFM Values)")
    profile = rfm.groupby("Segment")[
        ["Recency","Frequency","Monetary"]
    ].mean().round(2)
    profile["Customer Count"] = rfm["Segment"].value_counts()
    st.dataframe(
        profile.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📦 RFM Spread by Segment")
    metric = st.selectbox("Select Metric",
                          ["Recency","Frequency","Monetary"])