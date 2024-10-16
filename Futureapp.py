import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler


# Download stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Prepare data for LSTM model
def prepare_data(data, look_back=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

# Create and train LSTM model
def create_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    return model

# Load or train model
def load_or_train_model(x_train, y_train, model_path="keras_model.h5"):
    if os.path.exists(model_path):
        print("Loading existing model...")
        return load_model(model_path)
    else:
        print("Training new model...")
        model = create_model(x_train, y_train)
        model.save(model_path)
        print("Model trained and saved successfully.")
        return model

# Make predictions for multiple days
def predict_prices_multi_day(model, data, scaler, look_back=100, days_to_predict=30):
    last_sequence = data[-look_back:].values.reshape(-1, 1)
    scaled_data = scaler.transform(last_sequence)
    
    predictions = []
    for _ in range(days_to_predict):
        seq = scaled_data[-look_back:]
        seq = seq.reshape(1, look_back, 1)
        pred = model.predict(seq)
        predictions.append(pred[0, 0])
        scaled_data = np.append(scaled_data, pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Streamlit app
def main():
    st.title("Stock Price Prediction App")
    
    # User input
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-02-23"))
    
    if st.button("Analyze and Predict"):
        # Get stock data
        df = get_stock_data(ticker, start_date, end_date)
        
        # Display data description
        st.subheader('Data Description')
        st.write(df.describe())
        
        # Plotting Closing values on a chart
        st.subheader('Closing Price vs Time chart')
        fig = plt.figure(figsize=(12,6))
        plt.plot(df.Close, label = 'Closing Price')
        plt.legend()
        st.pyplot(fig)
        
        # Plotting with 100-day Moving Average
        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(df.Close, label = 'Closing Price')
        plt.plot(ma100, label = 'Moving Average of Prev 100 days')
        plt.legend()
        st.pyplot(fig)
        
        # Plotting with 100-day and 200-day Moving Averages
        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma100 = df.Close.rolling(100).mean()
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(df.Close, label = 'Closing Price')
        plt.plot(ma100, label = 'Moving Average of Prev 100 days')
        plt.plot(ma200, label = 'Moving Average of Prev 200 days')
        plt.legend()
        st.pyplot(fig)
        
        # Prepare data for training
        x_train, y_train, scaler = prepare_data(df.Close)
        
        # Load or train model
        model = load_or_train_model(x_train, y_train)
        
        # Make predictions for the next 30 days
        days_to_predict = 30
        predicted_prices = predict_prices_multi_day(model, df.Close, scaler, days_to_predict=days_to_predict)
        
        # Create future dates for predictions
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict)
        
        # Display prediction results
        st.subheader('Price Predictions')
        st.write(f"Last closing price: ${df.Close.iloc[-1]:.2f}")
        st.write(f"Predicted price after {days_to_predict} days: ${predicted_prices[-1]:.2f}")
        
        # Plot the stock prices with predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df.Close, label="Historical Prices")
        ax.plot(future_dates, predicted_prices, label="Predicted Prices", color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.title(f"{ticker} Stock Price Prediction")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
