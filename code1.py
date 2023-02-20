import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from tensorflow import keras
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2022-12-31'

st.title('Stock Trend Prediction')

userinput = st.text_input('Enter Stock Ticker: ','AAPL')
df = data.DataReader(userinput,'yahoo',2010-12-31,2022-12-31)


#Describing Data
st.subheader('Data from 2019-2022')
st.write(df.describe())


#Visualizations
st.subheader('Closing price vs Time chart ')
fig = plt.figure(figsize = (12,6))
plt.plot(df.close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA ')
ma100 = df.close.rolling(100).mean() 
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA & 200MA ')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.close)
st.pyplot(fig)


#Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])             #70 % data is training data      
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])  #30%is testing data


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training) 


#splitting data in xtrain and ytrain
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:1])
  y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train) ,np.array(y_train)


#Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input.data=scalar.fit_transform(final_df)

x_train = []
y_train = []


for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:1])
  y_train.append(data_training_array[i,0])
  
x_train, y_train = np.array(x_train) ,np.array(y_train)
y_predicated = modal.predict(x_test)
scalar = scalar.scale_


#ML Modal

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

modal = Sequential()
modal.add(LSTM(units=50, activation='relu',return_sequences = True,
               input_shape = (x_train.shape[1],1)))
modal.add(Dropout(0.2))

modal.add(LSTM(units=60, activation='relu',return_sequences = True))
modal.add(Dropout(0.3))

modal.add(LSTM(units=80, activation='relu',return_sequences = True))
modal.add(Dropout(0.4))


modal.add(LSTM(units=120, activation='relu',return_sequences = True))
modal.add(Dropout(0.5))


modal.add(Dense(units = 1))
modal.summery()
modal.compile(optimize='adam',loss='mean_squared_error')
modal.fit(x_train, y_train,epochos=50)



#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data=scalar.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)



#making Predications

y_predicted = modal.predicat(x_test)
scalar = scalar.scale_




scale_factor = 1/scalar[0]        
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor





#final Graph
st.subheader("Predictions Vs Original")
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
plt.show()
