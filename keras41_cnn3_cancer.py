import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# (x_train, y_train), (x_test, y_test) = mnist.load_data() 다르다


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #(455, 30)
print(y_train.shape) #(455,)
print(x_test.shape)  #(114,30)
print(y_test.shape)  #(114,)

x_train = x_train.reshape(455, 30, 1, 1)
x_test = x_test.reshape(114, 30, 1, 1)

# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder()
# onehotencoder.fit(y_train)
# y_train = onehotencoder.transform(y_train).toarray()
# y_test = onehotencoder.transform(y_test).toarray()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same',strides=1, input_shape=(30,1,1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Flatten()) # Flatten 중요 
model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)
print('loss, acc = ', loss, acc)

# loss, acc =  0.23169560730457306 0.6301531195640564 early_stopping
# loss, acc =  0.4818272888660431 0.9473684430122375 - flatten