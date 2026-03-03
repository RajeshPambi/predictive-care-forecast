📊 Predictive Forecasting of Care Load & Placement Demand

Time-Series Forecasting + Machine Learning for Capacity Planning
Built with Python, ARIMA, Random Forest & Streamlit

🚀 Project Overview

This project develops a predictive analytics system to forecast fluctuations in care load and placement demand using historical time-series data.

The system supports proactive planning and resource allocation by combining:

📈 Statistical Forecasting (ARIMA)

🌲 Machine Learning (Random Forest Regression)

📊 Exploratory Data Analysis (EDA)

🌐 Interactive Streamlit Dashboard

🎯 Problem Statement

Care systems operate in highly uncertain environments where sudden demand spikes can strain capacity.

This project aims to:

Identify trends and seasonality

Forecast future care load

Compare statistical vs ML approaches

Provide actionable insights via a dashboard

🛠️ Tech Stack

Python 3.x

Pandas

NumPy

Matplotlib

Scikit-learn

Statsmodels

Streamlit

📂 Project Structure
predictive-care-forecast/
│
├── data/
│   └── uac_data.csv
│
├── 3_streamlit_app.py
│
├── requirements.txt
└── README.md
📊 Dataset Information

The dataset contains daily time-series records including:

Date

Children apprehended and placed in custody

Children in custody

Children transferred out

Children in care

Children discharged

Target Variable:

Children in HHS Care

🔍 Exploratory Data Analysis

EDA includes:

Time-series trend visualization

Weekly seasonal decomposition

Correlation matrix

Summary statistics

Missing value analysis

Key Findings:

Clear seasonal patterns observed

Strong correlation between custody inflow and care load

Long-term structural trend visible

🤖 Modeling Approach
1️⃣ ARIMA (Baseline Model)

Captures temporal dependencies

Used as statistical benchmark

2️⃣ Random Forest Regressor

Captures non-linear relationships

Utilizes:

Lag features (t-1, t-7, t-14)

Rolling averages

Date-based features

📈 Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

Model comparison performed to determine best predictive performance.

🌐 Streamlit Dashboard

The interactive dashboard provides:

Dataset overview

Trend & seasonal visualization

Forecast plots

Model performance metrics

Real-time predictions

▶ Run Locally
git clone https://github.com/your-username/predictive-care-forecast.git
cd predictive-care-forecast
pip install -r requirements.txt
streamlit run 3_streamlit_app.py

Open in browser:

  Local URL: http://localhost:8501
  Network URL: http://10.207.172.221:8501





















## Author
Pambi Rajesh