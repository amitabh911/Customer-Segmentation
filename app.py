# ============================================================
# app.py — CUSTOMER SEGMENTATION STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import joblib

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Load Data & Models ───────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect("retail.db")
    rfm  = pd.read_sql("SELECT * FROM rfm_clusters", conn)
    pred = pd.read_sql("SELECT * FROM predictions",  conn)
    conn.close()
    return rfm, pred

@st.cache_resource
def load_models():
    scaler    = joblib.load("models/scaler.pkl")
    model     = joblib.load("models/best_classifier.pkl")
    return scaler, model

rfm, predictions = load_data()
scaler, model    = load_models()

# Segment → color mapping
COLORS = {
    "🏆 Champions"          : "#FFD700",
    "🌱 New/Recent Customers": "#90EE90",
    "⚠️ At Risk"            : "#FF6347",
    "💤 Lost Customers"     : "#A9A9A9"
}

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
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

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers",  f"{len(rfm):,}")
    col2.metric("Avg Recency",      f"{rfm['Recency'].mean():.0f} days")
    col3.metric("Avg Frequency",    f"{rfm['Frequency'].mean():.1f} orders")
    col4.metric("Avg Monetary",     f"£{rfm['Monetary'].mean():.2f}")

    st.markdown("---")

    # RFM Distributions
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

    # 3D Scatter
    st.subheader("🌐 Customers in 3D RFM Space")
    fig = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
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

    # Pie chart
    col1, col2 = st.columns(2)

    with col1:
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.pie(
            seg_counts, values="Count", names="Segment",
            title="Segment Distribution",
            color="Segment",
            color_discrete_map=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            seg_counts, x="Segment", y="Count",
            title="Customer Count per Segment",
            color="Segment",
            color_discrete_map=COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Segment Profiles
    st.subheader("📋 Segment Profiles (Mean RFM Values)")
    profile = rfm.groupby("Segment")[
        ["Recency","Frequency","Monetary"]
    ].mean().round(2)
    profile["Customer Count"] = rfm["Segment"].value_counts()
    st.dataframe(profile.style.background_gradient(cmap="Blues"), 
                 use_container_width=True)

    st.markdown("---")

    # RFM Boxplots by Segment
    st.subheader("📦 RFM Spread by Segment")
    metric = st.selectbox("Select Metric", ["Recency","Frequency","Monetary"])
    fig = px.box(
        rfm, x="Segment", y=metric,
        color="Segment",
        title=f"{metric} by Segment",
        color_discrete_map=COLORS
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER LOOKUP
# ════════════════════════════════════════════════════════════
elif page == "🔍 Customer Lookup":

    st.title("🔍 Customer Lookup")
    st.markdown("---")

    # Filter by Segment
    selected_segment = st.selectbox(
        "Filter by Segment",
        ["All"] + list(rfm["Segment"].unique())
    )

    if selected_segment == "All":
        filtered = rfm
    else:
        filtered = rfm[rfm["Segment"] == selected_segment]

    st.markdown(f"**Showing {len(filtered):,} customers**")
    st.dataframe(
        filtered[["CustomerID","Recency","Frequency",
                  "Monetary","Segment"]].reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")

    # Individual Customer Search
    st.subheader("🔎 Search by Customer ID")
    customer_id = st.number_input("Enter Customer ID",
                                   min_value=int(rfm["CustomerID"].min()),
                                   max_value=int(rfm["CustomerID"].max()),
                                   step=1)

    if st.button("Search"):
        result = rfm[rfm["CustomerID"] == customer_id]
        if len(result) > 0:
            r = result.iloc[0]
            st.success(f"Customer Found!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recency",   f"{r['Recency']} days")
            c2.metric("Frequency", f"{r['Frequency']} orders")
            c3.metric("Monetary",  f"£{r['Monetary']:.2f}")
            c4.metric("Segment",   r["Segment"])
        else:
            st.error("Customer ID not found.")


# ════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT SEGMENT
# ════════════════════════════════════════════════════════════
elif page == "🤖 Predict Segment":

    st.title("🤖 Predict Customer Segment")
    st.markdown("Enter a new customer's RFM values to predict their segment.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency   = st.number_input("Recency (days since last purchase)",
                                     min_value=1, max_value=365, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of orders)",
                                     min_value=1, max_value=200, value=5)
    with col3:
        monetary  = st.number_input("Monetary (total spend £)",
                                     min_value=1, max_value=50000, value=500)

    st.markdown("---")

    if st.button("🔮 Predict Segment", use_container_width=True):

        # Build input
        new_customer = pd.DataFrame({
            "Recency"   : [recency],
            "Frequency" : [frequency],
            "Monetary"  : [monetary]
        })

        # Transform
        new_log            = np.log1p(new_customer)
        new_log.columns    = ["R_log", "F_log", "M_log"]
        new_scaled         = scaler.transform(new_log)

        # Predict
        cluster_to_segment = rfm.drop_duplicates("KMeans_Cluster").set_index(
            "KMeans_Cluster")["Segment"].to_dict()
        predicted_cluster  = model.predict(new_scaled)[0]
        predicted_segment  = cluster_to_segment[predicted_cluster]

        # Display result
        color = COLORS.get(predicted_segment, "#ffffff")
        st.markdown(f"""
        <div style='background-color:{color}; padding:30px;
                    border-radius:12px; text-align:center;'>
            <h1 style='color:black;'>{predicted_segment}</h1>
            <h3 style='color:black;'>Cluster {predicted_cluster}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Business recommendation
        st.subheader("💡 Recommended Action")
        actions = {
            "🏆 Champions"           : "Reward with loyalty points, early access to new products, and exclusive offers. These are your most valuable customers — keep them engaged.",
            "🌱 New/Recent Customers" : "Send a welcome series, offer a discount on second purchase, and educate them about your product range.",
            "⚠️ At Risk"             : "Send a win-back email campaign with a special discount. Act quickly before they leave permanently.",
            "💤 Lost Customers"      : "Try one last re-engagement campaign. If no response, focus budget on other segments."
        }
        st.info(actions.get(predicted_segment, "No recommendation available."))