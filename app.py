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
st.title("Stock Price Predictor")


start = st.date_input("Start Date", value=pd.to_datetime('2010-01-01'))
end = st.date_input("End Date", value=pd.to_datetime('2024-10-01'))
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
st.markdown("[Find Tickers](https://finance.yahoo.com)")
predict_button = st.button('Analyze & Predict')

if predict_button:
    df = data.data.get_data_yahoo(user_input, start, end)
    # Displaying data description
    # st.subheader(f'Data from {start} to {end}')
    # st.write(df.describe())

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

    # Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load Model
    model = load_model('keras_model.h5')

    # Testing Part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Making Predictions
    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Final Prediction vs Original Graph
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
