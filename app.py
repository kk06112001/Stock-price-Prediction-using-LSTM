import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = tf.keras.models.load_model('lstm_stock_price.h5')

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def download_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']]
    return data

def prepare_data(data, time_step=60):
    scaled_data = scaler.transform(data[['Close']])
    
    X_lstm = []
    y_lstm = []

    for i in range(time_step, len(scaled_data)):
        X_lstm.append(scaled_data[i-time_step:i, 0])
        y_lstm.append(scaled_data[i, 0])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    # Reshape for LSTM input
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    return X_lstm, y_lstm

st.title('Stock Price Prediction using LSTM')
stock_symbol = st.text_input('Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):', 'AAPL')
start_date = st.date_input('Start Date:', pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date:', pd.to_datetime('2023-12-31'))

data = download_data(stock_symbol, start_date, end_date)
X_lstm, _ = prepare_data(data)

predictions_lstm = model.predict(X_lstm)

predictions_lstm = scaler.inverse_transform(predictions_lstm)

st.write('### Stock Price Prediction')
plt.figure(figsize=(10, 6))
plt.plot(data['Close'].values[-len(predictions_lstm):], label='Actual Prices')
plt.plot(predictions_lstm, label='Predicted Prices')
plt.legend()
plt.title('LSTM Stock Price Prediction')
st.pyplot()

if st.checkbox('Show raw data'):
    st.write(data)

st.write(f"Predicted Stock Price for {stock_symbol} on {end_date}: ${predictions_lstm[-1][0]:.2f}")
