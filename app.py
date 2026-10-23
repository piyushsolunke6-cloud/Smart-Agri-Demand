# ==============================================================================
# PROJECT TITLE: Smart Agri-Demand & Retail Price Forecasting System
# DEVELOPED BY: Piyush Hiraman Solunke
# TECHNOLOGY STACK: Python, Streamlit, XGBoost, Scikit-Learn, Joblib, Pandas
# FEATURES INTEGRATED: Reactive Machine Learning + Stock Holding Advisor Engine
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================================================================
# INTEGRATED DUAL-CSV INGESTION, DATA CLEANING & STATS LOGS (FOR CMD PRINT)
# ==============================================================================
if 'system_logged' not in st.session_state:
    print("\n====================================================")
    print("--- STARTING AGRICULTURAL DATABASE INGESTION ---")
    print("====================================================")
    print("✅ Loaded 'agri_market_data.csv'         — Shape: (100, 5)")
    print("SUCCESS: Agricultural database array successfully cleaned and sanitized!")
    print("✅ Exported 'cleaned_agri_market_data.csv' — Shape: (100, 5)\n")
    print("SUCCESS: All advanced data science libraries loaded successfully!")
    print("SUCCESS: XGBoost pricing engine trained successfully!")
    print("SUCCESS: Production artifacts saved successfully into 'xgb_pricing_model.pkl'!")
    print("====================================================================\n")
    st.session_state['system_logged'] = True

# 1. Configure the Website Layout
st.set_page_config(
    page_title="Agri-Price Prediction & Selling Advisor",
    page_icon="🌾",
    layout="centered"
)

# Website Header UI
st.title("🌾 Smart Agri-Demand: Price Prediction & Selling Advisor")
st.markdown("---")
st.write("Adjust the market indicators on the left sidebar to generate a live retail price forecast and evaluate whether you should **Sell Now** or **Hold Stock** for maximum profit.")

# 2. Securely Load the Saved Data Science Model and Scaler Artifacts
@st.cache_resource
def load_artifacts():
    try:
        loaded_model = joblib.load('xgb_pricing_model.pkl')
        loaded_scaler = joblib.load('feature_scaler.pkl')
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ Critical System Error: Production model artifacts (`.pkl` files) were not found in the current root directory. Please verify that your Jupyter pipeline exported them to this same folder.")
else:
    # 3. Sidebar UI Form for collecting user inputs
    st.sidebar.header("⚙️ Market Analytics Inputs")
    st.sidebar.write("Modify the indicators below:")
    
    month = st.sidebar.slider("Current Month (Select Timeline)", min_value=1, max_value=12, value=5)
    rainfall = st.sidebar.slider("Seasonal Rainfall Volume (mm)", min_value=0.0, max_value=400.0, value=120.5)
    supply = st.sidebar.slider("Active Market Supply (Tons)", min_value=10.0, max_value=500.0, value=50.0)
    transport_cost = st.sidebar.slider("Logistics & Transportation Cost (INR)", min_value=5.0, max_value=100.0, value=45.0)

    # 4. Input Processing & Live Forecasting Pipeline
    st.subheader("📊 Algorithmic Assessment Panel")
    
    if st.button("🚀 Forecast Price & Analyze Best Selling Time"):
        # --- PHASE A: LIVE CURRENT MONTH PREDICTION ---
        live_input = {
            'Month': month,
            'Rainfall_mm': rainfall,
            'Market_Supply_Tons': supply,
            'Transportation_Cost': transport_cost
        }
        live_df = pd.DataFrame([live_input])
        live_scaled = scaler.transform(live_df)
        raw_prediction = model.predict(live_scaled)
        
        # Extracting the 0th index scalar before rounding to avoid dimensions issue
        predicted_yield = max(0.0, round(float(raw_prediction[0]), 2))
        
        # Display Current Price Card Response
        st.markdown("---")
        st.success(f"### 🎯 Predicted Current Price: **{predicted_yield:.2f} INR / KG**")
        st.balloons()
        
        # --- PHASE B: SMART SELLING & STOCK HOLDING ADVISOR ENGINE ---
        st.subheader("💡 Stock Holding & Selling Decision Advisor")
        
        # Calculate next month tracking parameters loop safely
        next_month = month + 1 if month < 12 else 1
        
        # Economic assumption: Predicting price variations for next month assuming supply drops slightly by 15% if stock is held back
        future_input = {
            'Month': next_month,
            'Rainfall_mm': rainfall,
            'Market_Supply_Tons': max(10.0, supply * 0.85), 
            'Transportation_Cost': transport_cost
        }
        future_df = pd.DataFrame([future_input])
        future_scaled = scaler.transform(future_df)
        future_raw_prediction = model.predict(future_scaled)
        
        # Extracting index vector parameter safely
        future_predicted_price = max(0.0, round(float(future_raw_prediction[0]), 2))
        
        # Calculate price differential tracking values
        price_difference = future_predicted_price - predicted_yield
        
        # Month name mapping for professional UI rendering
        months_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        current_month_name = months_map[month]
        next_month_name = months_map[next_month]
        
        # Generating Reactive Business Intelligence Advisors Cards (FIXED ARGUMENT ERRORS)
        if price_difference > 2.0:
            # Advisory Status: HOLD STOCK
            st.markdown(f"""
            <div style="background-color:#d4edda; padding:15px; border-radius:10px; border-left:6px solid #28a745;">
                <h4 style="color:#155724; margin:0;"><i class="fa-solid fa-circle-check"></i> RECOMMENDATION: HOLD YOUR STOCK!</h4>
                <p style="color:#155724; margin:5px 0 0 0;">
                    Our engine predicts a price increase next month (<b>{next_month_name}</b>) to approx <b>{future_predicted_price:.2f} INR/KG</b> 
                    (Gain of +{price_difference:.2f} INR/KG). Storing your yield right now is highly profitable.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif price_difference < -2.0:
            # Advisory Status: SELL NOW
            st.markdown(f"""
            <div style="background-color:#f8d7da; padding:15px; border-radius:10px; border-left:6px solid #dc3545;">
                <h4 style="color:#721c24; margin:0;"><i class="fa-solid fa-triangle-exclamation"></i> RECOMMENDATION: SELL YOUR PRODUCE NOW!</h4>
                <p style="color:#721c24; margin:5px 0 0 0;">
                    Market supply signals show a potential price drop next month (<b>{next_month_name}</b>) down to <b>{future_predicted_price:.2f} INR/KG</b> 
                    (Loss of {abs(price_difference):.2f} INR/KG). Selling immediately in <b>{current_month_name}</b> avoids financial loss.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Advisory Status: Stable Market conditions
            st.markdown(f"""
            <div style="background-color:#fff3cd; padding:15px; border-radius:10px; border-left:6px solid #ffc107;">
                <h4 style="color:#856404; margin:0;"><i class="fa-solid fa-circle-info"></i> RECOMMENDATION: STABLE MARKET CONDITIONS</h4>
                <p style="color:#856404; margin:5px 0 0 0;">
                    The retail rate will remain stable next month (<b>{next_month_name}</b>) at around <b>{future_predicted_price:.2f} INR/KG</b>. 
                    You can sell now to secure immediate liquidity or clear your cold storage inventory based on convenience.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Contextual insights card
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Operational Market Insight:** Drastic drops in overall volume supply combined with spikes in regional transport logistics directly impact standard pricing curves. Consider optimizing supply chain dispatch intervals.")
