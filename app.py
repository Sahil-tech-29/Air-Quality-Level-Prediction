import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib


st.set_page_config(page_title="Air Quality Level Prediction", layout="centered")
SEQ_LEN = 30

# I load the model , scaler and data  
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("project_final_dataset_obtained.xlsx")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df


model = load_lstm_model()
scaler = load_scaler()
df = load_data()

# So , my model , scaler and dataset is loaded to memory now  

# Functions that are used to show category of aqi and what health advisory to give based on aqi value
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    else:
        return "Severe"

def health_advisory(aqi):
    if aqi <= 50:
        return "Air quality is satisfactory. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Acceptable air quality. Sensitive people should take care."
    elif aqi <= 200:
        return "Limit prolonged outdoor exertion."
    elif aqi <= 300:
        return "Avoid outdoor activities. Wear a mask if necessary."
    else:
        return "Health alert! Stay indoors and follow precautions."


# and here , our web app starts 
st.title("Air Quality Level Prediction System")
st.markdown("""
This application predicts **future AQI levels** using  
**LSTM-based time series forecasting**.

📍 Location: **New Delhi**  
📅 Data Period: **2022 – 2024**
""")

# we set the tiem line for our prediction
# that is min date must be 29 days less than selected date
# and the maximum date that can be used for prediction is the last date of 2024 
min_required_date = df['DateTime'].iloc[SEQ_LEN - 1]
max_date = df['DateTime'].max()

# similar information is shown on web app by using info() 
st.info(
    f"LSTM requires at least **{SEQ_LEN} days** of historical data. "
    f"Select dates from **{min_required_date.date()}** onwards."
)

# Here i wrote the logic of selecting the date by user
selected_date = st.date_input(
    "📅 Select a reference date",
    min_value=min_required_date.date(),
    max_value=max_date.date()
)

# Now , i made a button for prediction , and here is the main logic of prediction when user clicks the predict button 
# Everything that will be shown under this button 
if st.button("🔮 Predict AQI for Next 7 Days"):

    # Past_data contains the data of last 30 days from selected date 
    past_data = df[df['DateTime'] < pd.to_datetime(selected_date)].tail(SEQ_LEN)

    # edge case - if user enters invalid date beyond boundaries , then show warning
    if len(past_data) < SEQ_LEN:
        st.warning("Not enough historical data for prediction.")
        st.stop()

    # then i scaled the values of aqi 
    past_scaled = scaler.transform(past_data[['AQI']])

    # the current sequence that will be trained by model 
    current_sequence = past_scaled.reshape(1, SEQ_LEN, 1)

    future_predictions_scaled = []

    # Now I have to predict the aqi for next 7 days 
    # this loop predicts the aqi for next 7 days from the date selected by user 
    for i in range(7):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions_scaled.append(next_pred[0, 0])
        current_sequence = np.append(
            current_sequence[:, 1:, :],
            next_pred.reshape(1, 1, 1),
            axis=1
        )

    # the aqi values predicted are appended in the list of future_prediction after inverse scaling those values 
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1)
    ).flatten()

    # Now we have too show the results in tabular form that must have date , predicted aqi and its category 
    last_actual_date = pd.to_datetime(selected_date)
    future_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1),
        periods=7,
        freq='D'
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates.date,
        "Forecasted AQI": future_predictions.round().astype(int)
    })
    forecast_df["AQI Category"] = forecast_df["Forecasted AQI"].apply(aqi_category)

    
    st.subheader("📊 7-Day AQI Forecast")
    st.dataframe(forecast_df, use_container_width=True)
    # Now , my table of prediction is formed 
    
    # Now , I want to show the worst day having the highest aqi from the predicted week 
    worst = forecast_df.loc[forecast_df["Forecasted AQI"].idxmax()]

    st.subheader("🩺 Health Advisory")
    st.info(
        f"⚠️ Highest predicted AQI is **{worst['Forecasted AQI']}** "
        f"on **{worst['Date']}**. {health_advisory(worst['Forecasted AQI'])}"
    )

    # SShowing the another factor that is Pollution Alert that will be shown as a metric
    st.subheader("🚨 Pollution Alert")
    st.metric(
        label=f"Worst AQI on {worst['Date']}",
        value=int(worst["Forecasted AQI"]),
        delta=aqi_category(worst["Forecasted AQI"])
    )

    # Now the main part that is display of graph  
    st.subheader("📈 AQI Trend & Forecast")


    actual_window = df[
        df['DateTime'] <= pd.to_datetime(selected_date)
    ].tail(30)

    actual_dates = actual_window['DateTime']
    actual_aqi = actual_window['AQI']

    plt.figure(figsize=(11, 5))

    plt.plot(
        actual_dates,
        actual_aqi,
        label="Actual AQI (Last 30 Days)",
        marker="o"
    )


    plt.plot(
        future_dates,
        future_predictions,
        label="Forecasted AQI (Next 7 Days)",
        linestyle="--",
        marker="o"
    )


    plt.axvline(
        x=pd.to_datetime(selected_date),
        linestyle=":",
        color="gray",
        label="Forecast Start"
    )

    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.title("AQI Trend & 7-Day Forecast (LSTM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    st.pyplot(plt)

    # At the end i want to show the weekly insight based on my prediction 
    avg_aqi = int(forecast_df["Forecasted AQI"].mean())

    if avg_aqi > 300:
        insight = "Air quality is expected to remain **severely polluted** this week."
    elif avg_aqi > 200:
        insight = "Persistent **poor air quality** is expected."
    else:
        insight = "Air quality shows **moderate improvement**."

    st.info(f"📌 Weekly Insight: {insight}")


    # Here , i inserted the download button to sownload my prediction as csv file 
    st.download_button(
        "⬇️ Download 7-Day AQI Forecast",
        data=forecast_df.to_csv(index=False),
        file_name="aqi_7_day_forecast.csv",
        mime="text/csv"
    )
