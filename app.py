import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import streamlit as st
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Override Yahoo Finance data reader
yf.pdr_override()

# Title of the app
st.title("Multivariate Stock Price Predictor")

start = st.date_input("Start Date", value=pd.to_datetime('2000-01-01'))
end = st.date_input("End Date", value=pd.to_datetime('2023-09-15'))
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
st.markdown("[Find Tickers](https://finance.yahoo.com)")

predict_button = st.button('Analyze & Predict')

if predict_button:
    df = data.data.get_data_yahoo(user_input, start, end)

    # Displaying data description
    st.subheader(f'Data from {start} to {end}')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.legend()
    st.pyplot(fig)

    # Plotting with 100-Day Moving Average
    st.subheader('Closing Price vs Time chart with 100 day Moving Average(MA)')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.plot(ma100, label='100-Day Moving Average')
    plt.legend()
    st.pyplot(fig)

    # Plotting with 100-Day and 200-Day Moving Averages
    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.plot(ma100, label='100-Day Moving Average')
    plt.plot(ma200, label='200-Day Moving Average')
    plt.legend()
    st.pyplot(fig)

    # Selecting features for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[features])

    # Create dataset with multiple features
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), :])
            Y.append(dataset[i + time_step, 3])  # 3 is the index of 'Close' in features
        return np.array(X), np.array(Y)

    # Prepare testing data
    time_step = 100
    X_test, y_test = create_dataset(df_scaled, time_step)

    # Load Model
    model = load_model('keras_model.h5')

    # Making Predictions
    y_predicted = model.predict(X_test)

    # Inverse transform to get actual prices
    y_predicted_actual = np.zeros((len(y_predicted), len(features)))
    y_predicted_actual[:, 3] = y_predicted.flatten()  
    y_predicted_actual = scaler.inverse_transform(y_predicted_actual)[:, 3]

    y_test_actual = np.zeros((len(y_test), len(features)))
    y_test_actual[:, 3] = y_test
    y_test_actual = scaler.inverse_transform(y_test_actual)[:, 3]

    # Final Prediction vs Original Graph
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, 'b', label='Original Price')
    plt.plot(y_predicted_actual, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
