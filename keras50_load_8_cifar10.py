import numpy as np
import tensorflow as tf

x_train = np.load('../data/npy/cifar10_x_train.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit(y_train)
y_train = onehotencoder.transform(y_train).toarray()
y_test = onehotencoder.transform(y_test).toarray()
# print(y_train.shape) #(50000,10)
# print(y_test.shape) #(10000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=500, kernel_size=(2,2), padding='same', strides=1, input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=20, callbacks=[early_stopping], validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=20)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_predict)

print('r2 = ', r2)