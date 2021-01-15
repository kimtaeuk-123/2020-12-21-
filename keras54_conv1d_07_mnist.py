import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   #(60000, 28, 28)
print(x_test.shape, y_test.shape)     #(10000, 28, 28)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(60000,49,16)/255.
x_test = x_test.reshape(10000,49,16)/255.

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#OneHotencoding
#직접하기
from sklearn.preprocessing import OneHotEncoder
onehotencder = OneHotEncoder()
onehotencder.fit(y_train)
y_train = onehotencder.transform(y_train).toarray()
y_test = onehotencder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling2D, Flatten, Dropout, LSTM 

model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(49,16), activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Flatten())
model.add(Dense(10))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=10, validation_split=0.5, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss, acc : ', loss, acc)

# loss, acc :  0.02578057534992695 0.9262999892234802
