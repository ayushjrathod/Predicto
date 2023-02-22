import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import streamlit as st


st.title('Stock Trend Prediction')

userinput = st.text_input('Enter Stock Ticker: ','GOOG')
#df = pdr.DataReader(userinput,'yahoo',start,end)

yf.pdr_override()

spy = pdr.get_data_yahoo(userinput, start='2010-1-1', end='2022-12-30')


#Describing Data
st.subheader('Data from 2010-2022')
st.write(spy.describe())


#Visualizations
st.subheader('Closing price vs Time chart ')
fig = plt.figure(figsize = (12,6))
plt.plot(spy.Close)
st.pyplot(fig)



st.subheader('Closing price vs Time chart with 100MA ')
ma100 = spy.Close.rolling(100).mean() 
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(spy.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA & 200MA ')
ma100 = spy.Close.rolling(100).mean()
ma200 = spy.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(spy.Close)
st.pyplot(fig)


#Splitting data into training and testing
data_training = pd.DataFrame(spy['Close'][0:int(len(spy)*0.70)])             #70 % data is training data      
data_testing = pd.DataFrame(spy['Close'][int(len(spy)*0.70): int(len(spy))])  #30%is testing data


from sklearn.preprocessing import MinMaxScaler,StandardScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training) 


#splitting data in xtrain and ytrain
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train) ,np.array(y_train)


'''

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


modal.add(Dense(units = 1))  #to connect above 4 models



modal.summary()
modal.compile(optimizer='adam',loss='mean_squared_error')
modal.fit(x_train, y_train,epochs=50)
modal.save("kerasmodal.h5")
'''


modal=load_model("/home/ayush/Documents/VS Code/PBL/kerasmodal.h5")

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data=scalar.fit_transform(final_df)



x_test = []
y_test = []


for i in range(100,data_training_array.shape[0]):
  x_test.append(data_training_array[i-100:i])
  y_test.append(data_training_array[i,0])


x_test, y_test = np.array(x_test) ,np.array(y_test)

#making Predications
y_predicted = modal.predict(x_test)
scaler = scalar.scale_

scale_factor = 1/scaler[0]        
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#y_predicted = y_predicted.reshape(2189,100)
#y_predicted = y_predicted.reshape(-1,)

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




