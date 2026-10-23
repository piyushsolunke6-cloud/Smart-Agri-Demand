# =============================================================================
# Smart Agri-Demand & Retail Price Forecasting System
# Developed by: Piyush Hiraman Solunke
# GES's Sir Dr. M. S. Gosavi Polytechnic Institute, Nashik
# =============================================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smart Agri | Price Forecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f5faf5 0%, #ffffff 45%); }
      [data-testid="stSidebar"] { background: #102d24; }
      [data-testid="stSidebar"] * { color: #f3fff7 !important; }
      .hero {
        padding: 1.45rem 1.6rem; border-radius: 18px;
        background: linear-gradient(115deg, #123b2d, #21764d);
        color: white; margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(16, 55, 39, .13);
      }
      .hero h1 { color: white; margin: 0; font-size: 2rem; }
      .hero p { color: #e2f5e8; margin: .45rem 0 0 0; }
      .section-card {
        background: white; border: 1px solid #e2eee5;
        border-radius: 14px; padding: 1rem 1.1rem;
      }
      .small-muted { color: #64776c; font-size: .9rem; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2eee5;
        padding: 14px 16px; border-radius: 13px;
      }
      .footer { color: #6b7e73; font-size: .82rem; text-align: center; padding: 1rem 0 .4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MONTHS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
FEATURES = ["Month", "Rainfall_mm", "Market_Supply_Tons", "Transportation_Cost"]
MODEL_PATH = Path("xgb_pricing_model.pkl")
SCALER_PATH = Path("feature_scaler.pkl")
DATA_PATH = Path("agri_market_data.csv")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_price(model, scaler, month, rainfall, supply, transport_cost):
    row = pd.DataFrame([{
        "Month": month,
        "Rainfall_mm": rainfall,
        "Market_Supply_Tons": supply,
        "Transportation_Cost": transport_cost,
    }], columns=FEATURES)
    scaled = scaler.transform(row)
    prediction = float(np.asarray(model.predict(scaled)).reshape(-1)[0])
    return max(0.0, prediction)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌾 Smart Agri")
    st.caption("Market inputs & forecast controls")
    st.markdown("---")
    month = st.selectbox(
        "Forecast month",
        options=list(MONTHS.keys()),
        format_func=lambda m: f"{m:02d} — {MONTHS[m]}",
        index=4,
    )
    rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.5, step=0.5)
    supply = st.slider("Market supply (tons)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
    transport_cost = st.slider("Transportation cost (₹)", min_value=5.0, max_value=100.0, value=45.0, step=1.0)
    st.markdown("---")
    run_forecast = st.button("🔎 Generate forecast", type="primary", use_container_width=True)
    st.caption("Forecasts are model estimates, not guaranteed market prices.")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>🌾 Smart Agri-Demand & Retail Price Forecasting</h1>
      <p>Data-driven retail price estimates and a month-ahead comparison for agricultural market planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "**Developed by:** Piyush Hiraman Solunke  &nbsp; · &nbsp; "
    "**Department:** Computer Engineering  &nbsp; · &nbsp; "
    "**Institute:** GES's Sir Dr. M. S. Gosavi Polytechnic Institute, Nashik",
    unsafe_allow_html=True,
)

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error("Model files could not be loaded. Check that `xgb_pricing_model.pkl` and `feature_scaler.pkl` are present in the app's root folder.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

# ---------- Overview metrics ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Selected month", MONTHS[month])
with col2:
    st.metric("Rainfall input", f"{rainfall:.1f} mm")
with col3:
    st.metric("Market supply", f"{supply:,.0f} tons")

st.markdown("### 📈 Price forecast dashboard")
st.write("Change the inputs in the sidebar, then select **Generate forecast** to run the saved machine-learning model.")

if run_forecast or "forecast_result" not in st.session_state:
    try:
        current_price = predict_price(model, scaler, month, rainfall, supply, transport_cost)
        next_month = 1 if month == 12 else month + 1
        # Scenario assumption used by this project: supply is reduced by 15% next month.
        future_supply = max(10.0, supply * 0.85)
        next_price = predict_price(model, scaler, next_month, rainfall, future_supply, transport_cost)
        st.session_state["forecast_result"] = {
            "current_price": current_price,
            "next_price": next_price,
            "next_month": next_month,
            "future_supply": future_supply,
            "inputs": {
                "Month": month,
                "Rainfall_mm": rainfall,
                "Market_Supply_Tons": supply,
                "Transportation_Cost": transport_cost,
            },
        }
    except Exception as exc:
        st.error("Forecast could not be generated. Confirm that the saved scaler expects the same four input features in the same order.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

result = st.session_state["forecast_result"]
current_price = result["current_price"]
next_price = result["next_price"]
next_month = result["next_month"]
difference = next_price - current_price

m1, m2, m3 = st.columns(3)
m1.metric(f"Estimated price · {MONTHS[month]}", f"₹{current_price:,.2f}/kg")
m2.metric(f"Scenario estimate · {MONTHS[next_month]}", f"₹{next_price:,.2f}/kg", delta=f"₹{difference:+,.2f}/kg")
m3.metric("Scenario supply", f"{result['future_supply']:,.1f} tons", delta="-15% assumed")

chart_df = pd.DataFrame({
    "Month": [MONTHS[month], MONTHS[next_month]],
    "Estimated price (₹/kg)": [current_price, next_price],
}).set_index("Month")
left, right = st.columns([1.25, 1])
with left:
    st.markdown("#### Price comparison")
    st.bar_chart(chart_df, use_container_width=True)
with right:
    st.markdown("#### Selling-time guidance")
    if difference > 2.0:
        st.success(f"**Model signal: Consider holding**\n\nThe next-month scenario is ₹{difference:.2f}/kg higher. Compare this possible gain with storage, spoilage, finance, and handling costs before deciding.")
    elif difference < -2.0:
        st.error(f"**Model signal: Consider selling sooner**\n\nThe next-month scenario is ₹{abs(difference):.2f}/kg lower. Check actual local mandi prices and selling costs before acting.")
    else:
        st.warning("**Model signal: Relatively stable**\n\nThe estimated change is small. Decide using local market quotes, quality, storage costs, and cash-flow needs.")
    st.caption("The next-month result is a scenario, not a guaranteed forecast: rainfall and transport cost are held constant, while supply is assumed to fall by 15%.")

with st.expander("🔢 Inputs used for this forecast"):
    st.dataframe(pd.DataFrame([result["inputs"]]), use_container_width=True, hide_index=True)

# ---------- Dataset preview ----------
st.markdown("---")
st.markdown("### 🗂️ Dataset preview")
if DATA_PATH.exists():
    try:
        data = pd.read_csv(DATA_PATH)
        st.caption(f"Loaded `{DATA_PATH.name}` · {data.shape[0]} rows · {data.shape[1]} columns")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
        if not data.empty:
            numeric_cols = data.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Explore a numeric column", numeric_cols)
                st.line_chart(data[selected_col], use_container_width=True)
    except Exception as exc:
        st.info("Dataset file exists, but it could not be displayed as a table.")
        with st.expander("Technical details"):
            st.code(str(exc))
else:
    st.info("For the dataset preview, add `agri_market_data.csv` to the app's root folder. Forecasting can still work with the model files.")

# ---------- About ----------
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        **Objective:** Estimate agricultural retail price using market and seasonal indicators.

        **Input features**
        - `Month`
        - `Rainfall_mm`
        - `Market_Supply_Tons`
        - `Transportation_Cost`

        **Model output:** Estimated retail price per kilogram (₹/kg).

        **Technology stack:** Python, Streamlit, Pandas, NumPy, Scikit-learn, XGBoost and Joblib.

        **Important limitation:** The selling-time panel compares the current estimate with a next-month scenario in which supply is reduced by 15%, while rainfall and transportation cost remain unchanged. It is a decision-support demonstration, not financial advice or a guarantee of future prices.
        """
    )

st.markdown(
    '<div class="footer">Smart Agri-Demand & Retail Price Forecasting System · Academic project demonstration</div>',
    unsafe_allow_html=True,
)
