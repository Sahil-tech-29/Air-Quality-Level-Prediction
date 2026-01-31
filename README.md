## 🌍 Air Quality Level Prediction System (LSTM-Based Time Series Forecasting)

An end-to-end deep learning application that forecasts Air Quality Index (AQI) levels for the next 7 days using historical air quality data and an LSTM-based time series model, with an interactive Streamlit dashboard for visualization and health advisories.

## 📌 Project Overview

Air pollution is a major public health concern, especially in metropolitan cities like New Delhi. Most traditional air quality monitoring systems only provide real-time AQI values, which limits early warnings and proactive planning.

This project focuses on predicting future AQI levels rather than only monitoring current values. By leveraging deep learning and historical data, the system provides 7-day AQI forecasts, air quality categorization, and health recommendations to support informed decision-making.

## 🎯 Objectives

Forecast AQI levels for the next 7 days

Capture temporal and seasonal pollution patterns using LSTM

Categorize AQI into standard air quality levels

Generate health advisories based on predicted AQI

Provide an interactive and user-friendly web interface

## 🧠 Methodology

Collect historical air quality, weather, and traffic-related data

Clean and preprocess data (handle missing values, normalization)

Convert AQI time series into supervised learning sequences

Train an LSTM model using 30-day sliding windows

Apply recursive forecasting for multi-step prediction

Visualize results and insights using Streamlit

## 🏗 System Architecture

Data Sources → Preprocessing → LSTM Model → AQI Forecasting → AQI Categorization & Health Advisory → Streamlit Dashboard

## ✨ Features

7-day AQI forecasting

LSTM-based time series prediction

AQI categorization (Good, Moderate, Poor, Very Poor, Severe)

Health advisories and alerts

Interactive graphs and tables

CSV report download

## 🛠 Tech Stack

Python

TensorFlow (LSTM)

Scikit-learn

Pandas, NumPy

Matplotlib

Streamlit

Joblib

## 📂 Project Structure
``` bash
Air-Quality-Prediction/
│
├── data/
│   └── air_quality_dataset.csv
│
├── model/
│   ├── lstm_model.h5
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
└── README.md
```
## ⚙ Installation & Setup

Clone the repository

```git clone <your-repo-link>
cd Air-Quality-Prediction
```

Create virtual environment (optional but recommended)

```python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)
```

Install dependencies
```
pip install -r requirements.txt
```

Run the application
```
streamlit run app.py
```

## 📊 Input

Historical AQI values (daily)

Last 30 days used as model input

## 📈 Output

Next 7 days AQI forecast

AQI category for each day

Health advisory messages

Visualization of trends

## 🧪 Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

## 🚀 Future Enhancements

Integrate real-time sensor data

Add meteorological forecasting inputs

Use hybrid deep learning models (CNN-LSTM)

Deploy on cloud platform

## 👤 Author

Sahil Bhardwaj
B.Tech CSE | Machine Learning & Deep Learning Enthusiast
