# ==========================================================
# PREDICTIVE FORECASTING DASHBOARD (ML VERSION)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/uac_data.csv")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    numeric_cols = [
        "Children in HHS Care",
        "Children transferred out of CBP custody",
        "Children discharged from HHS Care"
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

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
        "📁 Data Analysis & EDA",
        "📊 Overview Dashboard",
        "📈 Trend & Seasonality",
        "🔮 ML Care Load Forecast",
        "🏥 Discharge Prediction",
        "⚠️ Early Warning Panel"
    ]
)

# ==========================================================
# HOME
# ==========================================================

# ==========================================================
# HOME (EXECUTIVE PROFESSIONAL VERSION)
# ==========================================================

if page == "🏠 Home":

    st.title("Predictive Forecasting of Care Load & Placement Demand")
    st.subheader("AI-Driven Decision Support for HHS Capacity Planning")

    st.markdown("---")

    # ------------------------------------------------------
    # PROJECT OVERVIEW
    # ------------------------------------------------------

    st.markdown("### Project Overview")

    st.write("""
    This project transforms historical UAC operational data into forward-looking 
    predictive intelligence. Using time-series analysis and machine learning, 
    the system forecasts future care demand, discharge capacity, and potential 
    system stress.
    
    The goal is to enable proactive planning rather than reactive crisis response.
    """)

    st.markdown("---")

    # ------------------------------------------------------
    # CURRENT SYSTEM SNAPSHOT
    # ------------------------------------------------------

    st.markdown("### Current System Snapshot")

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Children in HHS Care",
        f"{int(latest['Children in HHS Care']):,}"
    )

    col2.metric(
        "Transfers to HHS",
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

    col4.metric(
        "Net Daily Pressure",
        f"{int(net_pressure):,}"
    )

    st.markdown("---")

    # ------------------------------------------------------
    # WHAT THIS SYSTEM DELIVERS
    # ------------------------------------------------------

    st.markdown("### What This System Delivers")

    st.write("""
    • 7–30 day forecasts of children in HHS care  
    • Discharge and transfer demand prediction  
    • Early warning signals for capacity stress  
    • Model comparison (Statistical vs Machine Learning)  
    • Confidence interval visualization  
    • Interactive executive dashboard  
    """)

    st.markdown("---")

    # ------------------------------------------------------
    # IMPACT STATEMENT
    # ------------------------------------------------------

    st.info("""
    By shifting from descriptive reporting to predictive forecasting,
    this system supports data-driven decisions that improve resource
    allocation, reduce overcrowding risk, and strengthen child-welfare outcomes.
    """)

# ==========================================================
# DATA ANALYSIS & AUTOMATED EDA
# ==========================================================

elif page == "📁 Data Analysis & EDA":

    st.title("Dataset Preview, Summary Statistics & Automated EDA")

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.markdown("## 📊 Dataset Preview")

    rows = st.slider("Select number of rows to preview", 5, 50, 10)
    st.dataframe(df.head(rows))

    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.markdown("---")

    # ------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------

    st.markdown("## 📈 Summary Statistics")

    st.dataframe(df.describe())

    st.markdown("### Missing Values")

    missing = df.isnull().sum()
    st.dataframe(missing)

    st.markdown("### Date Range")

    st.write(f"Start Date: {df.index.min().date()}")
    st.write(f"End Date: {df.index.max().date()}")

    st.markdown("---")

    # ------------------------------------------------------
    # AUTOMATED EDA VISUALS
    # ------------------------------------------------------

    st.markdown("## 🤖 Automated Exploratory Data Analysis")

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()

    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im)
    st.pyplot(fig)

    # Distribution Plots
    st.markdown("### Distribution Analysis")

    column_choice = st.selectbox(
        "Select column for distribution analysis",
        df.columns
    )

    fig2, ax2 = plt.subplots()
    df[column_choice].hist(bins=30, ax=ax2)
    ax2.set_title(f"Distribution of {column_choice}")
    st.pyplot(fig2)

    # Rolling Trend Visualization
    st.markdown("### 7-Day Rolling Trend")

    fig3, ax3 = plt.subplots()
    rolling = df[column_choice].rolling(7).mean()
    ax3.plot(df[column_choice], alpha=0.4, label="Original")
    ax3.plot(rolling, linewidth=2, label="7-Day Rolling Mean")
    ax3.legend()
    st.pyplot(fig3)
# ==========================================================
# OVERVIEW DASHBOARD
# ==========================================================

elif page == "📊 Overview Dashboard":

    st.title("Overview Dashboard")

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Children in HHS Care",
                f"{int(latest['Children in HHS Care']):,}")

    col2.metric("Transfers to HHS",
                f"{int(latest['Children transferred out of CBP custody']):,}")

    col3.metric("Discharges",
                f"{int(latest['Children discharged from HHS Care']):,}")

    net_pressure = (
        latest["Children transferred out of CBP custody"]
        - latest["Children discharged from HHS Care"]
    )

    col4.metric("Net Pressure", f"{int(net_pressure):,}")

    st.subheader("Care Load Over Time")

    fig, ax = plt.subplots()
    ax.plot(df["Children in HHS Care"])
    st.pyplot(fig)

# ==========================================================
# TREND & SEASONALITY
# ==========================================================

elif page == "📈 Trend & Seasonality":

    st.title("Trend & Seasonality Analysis")

    # Rolling Trend
    st.subheader("7-Day Rolling Trend")

    trend = df["Children in HHS Care"].rolling(7).mean()

    fig, ax = plt.subplots()
    ax.plot(df["Children in HHS Care"], alpha=0.4, label="Original")
    ax.plot(trend, linewidth=2, label="Trend")
    ax.legend()
    st.pyplot(fig)

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition")

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
# ML CARE LOAD FORECAST
# ==========================================================

# ==========================================================
# ML CARE LOAD FORECAST (PROFESSIONAL VERSION)
# ==========================================================

elif page == "🔮 ML Care Load Forecast":

    st.title("Advanced Machine Learning Care Load Forecast")

    horizon = st.selectbox("Forecast Horizon (Days)", [7, 14, 30])

    # ------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------

    data = df.copy()

    # Lag Features
    for lag in range(1, 15):
        data[f"lag_{lag}"] = data["Children in HHS Care"].shift(lag)

    # Rolling Statistics
    data["rolling_mean_7"] = data["Children in HHS Care"].rolling(7).mean()
    data["rolling_std_7"] = data["Children in HHS Care"].rolling(7).std()

    # Date Features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month

    data = data.dropna()

    feature_cols = [col for col in data.columns if col != "Children in HHS Care"]

    X = data[feature_cols]
    y = data["Children in HHS Care"]

    # ------------------------------------------------------
    # TIME-BASED TRAIN TEST SPLIT
    # ------------------------------------------------------

    split = int(len(data) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ------------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------------

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    st.subheader("Model Performance (Out-of-Sample)")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")

    # ------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------

    st.subheader("Key Forecast Drivers")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False).head(10)

    fig_imp, ax_imp = plt.subplots()
    importance.plot(kind="barh", ax=ax_imp)
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    # ------------------------------------------------------
    # RECURSIVE FORECASTING
    # ------------------------------------------------------

    last_row = data.iloc[-1:].copy()
    future_preds = []

    for _ in range(horizon):

        pred = model.predict(last_row[feature_cols])[0]
        future_preds.append(pred)

        # Shift lag features
        for lag in range(14, 1, -1):
            last_row[f"lag_{lag}"] = last_row[f"lag_{lag-1}"]

        last_row["lag_1"] = pred

        # Update rolling mean approx
        last_row["rolling_mean_7"] = np.mean(
            [last_row[f"lag_{i}"].values[0] for i in range(1, 8)]
        )

    future_dates = pd.date_range(
        start=df.index[-1],
        periods=horizon + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame(
        {"Forecast": future_preds},
        index=future_dates
    )

    # ------------------------------------------------------
    # CONFIDENCE BAND (Approximation)
    # ------------------------------------------------------

    residual_std = np.std(y_test - y_pred)

    forecast_df["Upper"] = forecast_df["Forecast"] + 1.96 * residual_std
    forecast_df["Lower"] = forecast_df["Forecast"] - 1.96 * residual_std

    # ------------------------------------------------------
    # PROFESSIONAL VISUALIZATION
    # ------------------------------------------------------

    st.subheader("Forecast Projection")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df["Children in HHS Care"],
        label="Historical",
        alpha=0.6
    )

    ax.plot(
        forecast_df["Forecast"],
        linestyle="--",
        linewidth=3,
        label="ML Forecast"
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["Lower"],
        forecast_df["Upper"],
        alpha=0.2
    )

    ax.axvline(df.index[-1], linestyle=":", linewidth=2)

    ax.set_title("Projected Children in HHS Care")
    ax.set_ylabel("Number of Children")
    ax.legend()

    st.pyplot(fig)

    # ------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------

    growth_projection = (
        forecast_df["Forecast"].iloc[-1] -
        df["Children in HHS Care"].iloc[-1]
    )

    st.subheader("Executive Insight")

    if growth_projection > 0:
        st.warning(
            f"Projected increase of {int(growth_projection):,} children "
            f"over next {horizon} days."
        )
    else:
        st.success(
            f"Projected decrease of {abs(int(growth_projection)):,} children "
            f"over next {horizon} days."
        )

# ==========================================================
# DISCHARGE PREDICTION
# ==========================================================
elif page == "🏥 Discharge Prediction":

    st.title("ML Discharge vs Transfer Forecast")

    horizon = 14

    def train_and_forecast(series_name):

        data = df.copy()

        # Create lag features
        for lag in range(1, 8):
            data[f"lag_{lag}"] = data[series_name].shift(lag)

        data = data.dropna()

        feature_cols = [f"lag_{lag}" for lag in range(1, 8)]

        X = data[feature_cols]
        y = data[series_name]

        split = int(len(data) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Recursive forecast
        last_values = data.iloc[-1][feature_cols].values.reshape(1, -1)

        future_preds = []

        for _ in range(horizon):
            next_pred = model.predict(last_values)[0]
            future_preds.append(next_pred)

            last_values = np.roll(last_values, 1)
            last_values[0][0] = next_pred

        return future_preds


    discharge_forecast = train_and_forecast(
        "Children discharged from HHS Care"
    )

    transfer_forecast = train_and_forecast(
        "Children transferred out of CBP custody"
    )

    future_dates = pd.date_range(
        start=df.index[-1],
        periods=horizon + 1,
        freq="D"
    )[1:]

    fig, ax = plt.subplots()

    ax.plot(future_dates, discharge_forecast, label="ML Discharges")
    ax.plot(future_dates, transfer_forecast, label="ML Transfers")

    ax.legend()
    st.pyplot(fig)

    imbalance = np.mean(transfer_forecast) - np.mean(discharge_forecast)

    if imbalance > 0:
        st.error("⚠ Capacity Stress Risk")
    else:
        st.success("System Stable")

# ==========================================================
# EARLY WARNING PANEL
# ==========================================================

elif page == "⚠️ Early Warning Panel":

    st.title("Early Warning Panel")

    growth = df["Children in HHS Care"].pct_change().iloc[-3:].mean()

    if growth > 0.03:
        st.error("🔴 HIGH RISK")
    elif growth > 0.01:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 LOW RISK")

    st.write("Recent Growth Rate:",
             round(growth * 100, 2), "%")