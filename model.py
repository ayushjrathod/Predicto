import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

# Splitting data into x_train and y_train
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping x_train if it's not already in the correct shape
if len(x_train.shape) == 2:
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, 
               input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu', return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)
model.save("keras_model.h5")
