# ==========================================================
# PREDICTIVE FORECASTING DASHBOARD (ML VERSION - FIXED)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="Predictive Care Forecast (ML)",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD + CLEAN DATA  ✅ FIXED
# ----------------------------------------------------------

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "data", "uac_data.csv")

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    # Convert numeric columns safely (remove commas if exist)
    numeric_cols = [
        "Children in HHS Care",
        "Children transferred out of CBP custody",
        "Children discharged from HHS Care"
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing rows
    df = df.dropna()

    return df


df = load_data()

# ----------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Overview Dashboard",
        "📈 Trend & Seasonality",
        "🔮 ML Care Load Forecast",
        "⚠️ Early Warning Panel"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("Predictive Forecasting of Care Load & Placement Demand")

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Children in HHS Care",
        f"{int(latest['Children in HHS Care']):,}"
    )

    col2.metric(
        "Transfers",
        f"{int(latest['Children transferred out of CBP custody']):,}"
    )

    col3.metric(
        "Discharges",
        f"{int(latest['Children discharged from HHS Care']):,}"
    )

    net_pressure = (
        latest["Children transferred out of CBP custody"]
        - latest["Children discharged from HHS Care"]
    )

    col4.metric("Net Daily Pressure", f"{int(net_pressure):,}")

# ==========================================================
# OVERVIEW DASHBOARD
# ==========================================================

elif page == "📊 Overview Dashboard":

    st.title("Overview Dashboard")

    fig, ax = plt.subplots()
    ax.plot(df["Children in HHS Care"])
    ax.set_title("Children in HHS Care Over Time")
    st.pyplot(fig)

# ==========================================================
# TREND & SEASONALITY
# ==========================================================

elif page == "📈 Trend & Seasonality":

    st.title("Trend & Seasonality")

    trend = df["Children in HHS Care"].rolling(7).mean()

    fig, ax = plt.subplots()
    ax.plot(df["Children in HHS Care"], alpha=0.4)
    ax.plot(trend, linewidth=2)
    st.pyplot(fig)

    try:
        decomposition = seasonal_decompose(
            df["Children in HHS Care"],
            model="additive",
            period=7
        )
        fig = decomposition.plot()
        st.pyplot(fig)
    except:
        st.warning("Not enough data for decomposition.")

# ==========================================================
# ML FORECAST
# ==========================================================

elif page == "🔮 ML Care Load Forecast":

    st.title("Machine Learning Forecast")

    horizon = st.selectbox("Forecast Horizon", [7, 14, 30])

    data = df.copy()

    # Lag features
    for lag in range(1, 8):
        data[f"lag_{lag}"] = data["Children in HHS Care"].shift(lag)

    data = data.dropna()

    feature_cols = [f"lag_{i}" for i in range(1, 8)]

    X = data[feature_cols]
    y = data["Children in HHS Care"]

    split = int(len(data) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.metric("MAE", f"{mae:,.2f}")
    st.metric("RMSE", f"{rmse:,.2f}")

    # Recursive forecast
    last_values = data.iloc[-1][feature_cols].values.reshape(1, -1)

    future_preds = []

    for _ in range(horizon):
        pred = model.predict(last_values)[0]
        future_preds.append(pred)

        last_values = np.roll(last_values, 1)
        last_values[0][0] = pred

    future_dates = pd.date_range(
        start=df.index[-1],
        periods=horizon + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame(
        {"Forecast": future_preds},
        index=future_dates
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Children in HHS Care"], label="Historical")
    ax.plot(forecast_df["Forecast"], linestyle="--", linewidth=3, label="Forecast")
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# EARLY WARNING PANEL
# ==========================================================

elif page == "⚠️ Early Warning Panel":

    st.title("Early Warning Panel")

    growth = (
        df["Children in HHS Care"]
        .pct_change()
        .iloc[-3:]
        .mean()
    )

    if growth > 0.03:
        st.error("🔴 HIGH RISK")
    elif growth > 0.01:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

    st.write("Recent Growth Rate:", round(growth * 100, 2), "%")
