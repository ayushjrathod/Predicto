#Importing necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

yf.pdr_override()

#Defining the start and end dates for the data frame
start = '2010-01-01'
end = '2023-02-23'

st.title("Stock Trend Prediction")

#Taking stock ticker from the user
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

#Scrapping Data from yahoo finance
df = data.data.get_data_yahoo(user_input, start, end)

#Describing Data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

#Visualizations

#Plotting Closing values on a chart
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Closing Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Closing Price')
plt.plot(ma100, label = 'Moving Average of Prev 100 days')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label = 'Closing Price')
plt.plot(ma100, label = 'Moving Average of Prev 100 days')
plt.plot(ma200, label = 'Moving Average of Prev 200 days')
plt.legend()
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #70% of the data frame is in training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) #Reamaining 30% of the data frame is in testing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
#This means that all the data in the Close column of the data frame will be scaled down between 0 to 1

data_training_array = scaler.fit_transform(data_training)

#Load Model
model = load_model('keras_model.h5')


#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


#Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)