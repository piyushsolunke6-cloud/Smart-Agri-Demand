from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="Smart Agri | Crop Price Prediction", page_icon="🌾", layout="wide")

st.title("🌾 Smart Agri-Demand & Retail Price Forecasting System")
st.caption("Crop-wise price prediction · Academic project demo")
st.write("**Developed by:** Piyush Hiraman Solunke · GES's Sir Dr. M. S. Gosavi Polytechnic Institute, Nashik")

DATA_PATH = Path("agri_market_data_100.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("agri_market_data.csv")

FEATURES_NUM = ["Month", "Rainfall_mm", "Market_Supply_Tons", "Transportation_Cost"]
CROP_COL = "Crop"
TARGET = "Retail_Price_per_KG"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def train_model(path, file_mtime):
    df = pd.read_csv(path)
    required = FEATURES_NUM + [CROP_COL, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    df = df[required].copy()
    df["Crop"] = df["Crop"].astype(str).str.strip().str.title()
    for col in FEATURES_NUM + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if len(df) < 10:
        raise ValueError("At least 10 complete dataset rows are needed to train the demo model.")
    X = df[FEATURES_NUM + [CROP_COL]]
    y = df[TARGET]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES_NUM),
            ("crop", OneHotEncoder(handle_unknown="ignore"), [CROP_COL]),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42, objective="reg:squarederror"))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": r2_score(y_test, pred) if len(y_test) > 1 else float("nan"),
        "test_rows": len(y_test),
    }
    return model, metrics, df

try:
    if not DATA_PATH.exists():
        st.error("Dataset file not found. Upload `agri_market_data_100.csv` to the same GitHub repository as this app.")
        st.stop()
    model, metrics, data = train_model(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as e:
    st.error("Dataset/model setup error: " + str(e))
    st.stop()

with st.sidebar:
    st.header("⚙️ Prediction Inputs")
    crop_options = sorted(data["Crop"].unique().tolist())
    crop = st.selectbox("🥕 Select Crop", crop_options)
    month = st.selectbox("Month", list(range(1, 13)), index=4,
                         format_func=lambda m: f"{m:02d} - {pd.Timestamp(2026, m, 1).strftime('%B')}")
    rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 120.0, 1.0)
    supply = st.slider("Market Supply (tons)", 10.0, 500.0, 100.0, 5.0)
    transport = st.slider("Transportation Cost (₹)", 5.0, 100.0, 15.0, 1.0)

input_df = pd.DataFrame([{
    "Month": month,
    "Rainfall_mm": rainfall,
    "Market_Supply_Tons": supply,
    "Transportation_Cost": transport,
    "Crop": crop,
}])
predicted_price = max(0.0, float(model.predict(input_df)[0]))

st.subheader("📊 Estimated Retail Price")
k1, k2, k3 = st.columns(3)
k1.metric("Selected crop", crop)
k2.metric("Predicted price", f"₹{predicted_price:.2f}/kg")
k3.metric("Dataset records", f"{len(data)}")

st.markdown("### 📈 Dataset graphs")
left, right = st.columns(2)
with left:
    st.markdown("**Retail price by crop**")
    st.bar_chart(data.groupby("Crop")[TARGET].mean())
with right:
    st.markdown("**Average price by month**")
    st.line_chart(data.groupby("Month")[TARGET].mean())

tab1, tab2, tab3 = st.tabs(["🗂️ Dataset Preview", "📉 Model Performance", "👨‍🎓 About Project"])
with tab1:
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="agri_market_data_100.csv", mime="text/csv")
with tab2:
    a, b, c = st.columns(3)
    a.metric("Test MAE", f"₹{metrics['mae']:.2f}/kg")
    b.metric("Test RMSE", f"₹{metrics['rmse']:.2f}/kg")
    c.metric("Test R²", f"{metrics['r2']:.3f}")
    st.caption(f"Metrics calculated on a held-out test set ({metrics['test_rows']} rows). The dataset is synthetic, so these scores are for demonstration only.")
with tab3:
    st.markdown("""
**Project:** Smart Agri-Demand & Retail Price Forecasting System

**Student:** Piyush Hiraman Solunke  
**Institute:** GES's Sir Dr. M. S. Gosavi Polytechnic Institute, Nashik

**Inputs:** Crop, Month, Rainfall, Market Supply, Transportation Cost  
**Target:** Retail Price per KG

**Technologies:** Python, Streamlit, Pandas, Scikit-learn, XGBoost

**Important:** The provided 100-row dataset is synthetic/illustrative, not verified real-market data. Predictions are educational examples, not guaranteed prices.
""")
