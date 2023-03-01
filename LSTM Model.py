#splitting data in xtrain and ytrain
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train) ,np.array(y_train)



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
