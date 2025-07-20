import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from prophet import Prophet

# Streamlit App Title
st.title("📈 Financial Forecasting Tool")
st.subheader("🔁 Upload your CSV file")

# File uploader
uploaded_file = st.file_uploader("Upload CSV (must have 'date' and 'revenue' columns)", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        if 'revenue' not in df.columns:
            st.error("❌ CSV must contain a 'revenue' column.")
        else:
            # Forecast method selection
            st.subheader("🧠 Choose Forecasting Method")
            method = st.radio("Select a forecasting method:", ["LSTM", "Prophet"])

            # Forecast period selection
            st.subheader("📅 Select Forecast Period")
            forecast_period = st.slider("Select number of months to forecast", 1, 12, 3)

            if method == "Prophet":
                # Prophet forecasting
                def forecast_revenue(df, periods=3):
                    df_prophet = df.rename(columns={"date": "ds", "revenue": "y"})
                    model = Prophet()
                    model.fit(df_prophet)

                    future = model.make_future_dataframe(periods=periods, freq='MS')
                    forecast = model.predict(future)
                    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                forecast_df = forecast_revenue(df, periods=forecast_period)
                forecast_df = forecast_df.tail(forecast_period)

                st.subheader("📊 Prophet Forecast Results")
                st.dataframe(forecast_df)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["date"], df["revenue"], label="Historical Revenue", marker='o')
                ax.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast", marker='x', color='green')
                ax.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2, color='green', label='Confidence Interval')
                ax.set_xlabel("Date")
                ax.set_ylabel("Revenue")
                ax.set_title("Prophet-Based Revenue Forecast")
                ax.legend()
                st.pyplot(fig)

                # Download
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast CSV", data=csv, file_name="prophet_forecast.csv", mime='text/csv')

            else:
                # LSTM forecasting
                data = df[['revenue']].values

                # Normalize
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)

                # Create sequences
                SEQ_LENGTH = 3
                def create_sequences(data, seq_length):
                    x, y = [], []
                    for i in range(len(data) - seq_length):
                        x.append(data[i:i+seq_length])
                        y.append(data[i+seq_length])
                    return np.array(x), np.array(y)

                x, y = create_sequences(scaled_data, SEQ_LENGTH)

                # LSTM Model
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(x, y, epochs=200, verbose=0)

                # Forecasting
                last_seq = scaled_data[-SEQ_LENGTH:]
                predictions = []
                current_seq = last_seq

                for _ in range(forecast_period):
                    pred = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
                    predictions.append(pred[0, 0])
                    current_seq = np.append(current_seq[1:], [[pred[0, 0]]], axis=0)

                # Inverse scale
                predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                forecast_dates = [df["date"].iloc[-1] + pd.DateOffset(months=i+1) for i in range(forecast_period)]

                forecast_df = pd.DataFrame({
                    "date": forecast_dates,
                    "forecasted_revenue": predicted_values
                })

                # Show forecast
                st.subheader("📊 LSTM Forecast Results")
                st.dataframe(forecast_df)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["date"], df["revenue"], label="Historical Revenue", marker='o')
                ax.plot(forecast_df["date"], forecast_df["forecasted_revenue"], label="Forecast", marker='x', color='green')
                ax.set_xlabel("Date")
                ax.set_ylabel("Revenue")
                ax.set_title("LSTM-Based Revenue Forecast")
                ax.legend()
                st.pyplot(fig)

                # Download
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast CSV", data=csv, file_name="lstm_forecast.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")
