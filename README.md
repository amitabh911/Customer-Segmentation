# 🛒 Customer Segmentation using RFM & Machine Learning

## Project Overview
An end-to-end customer segmentation project using real UK e-commerce 
data. Customers are segmented using RFM analysis and clustering 
algorithms, with a predictive classifier and interactive dashboard.

## Tech Stack
- Python, Pandas, NumPy
- SQLite (SQL queries for EDA and RFM)
- Scikit-learn, XGBoost, LightGBM
- Matplotlib, Seaborn, Plotly
- Streamlit (interactive dashboard)

## Project Structure
customer_segmentation/
├── notebook.ipynb       ← Main analysis notebook
├── app.py               ← Streamlit dashboard
├── retail.db            ← SQLite database
├── README.md
└── models/
    ├── scaler.pkl
    ├── kmeans_model.pkl
    └── best_classifier.pkl

## Phases
- Phase 0 — Setup & Database
- Phase 1 — SQL EDA
- Phase 2 — Python EDA
- Phase 3 — RFM Feature Engineering
- Phase 4 — Clustering (KMeans, Hierarchical, DBSCAN)
- Phase 5 — Predictive Classifier
- Phase 6 — Streamlit Dashboard

## Segments Identified
| Segment | Description |
|---|---|
| 🏆 Champions | High value, frequent, recent buyers |
| 🌱 New Customers | Recent but low frequency |
| ⚠️ At Risk | Used to buy but gone quiet |
| 💤 Lost | Low recency, frequency, monetary |

## How to Run
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py