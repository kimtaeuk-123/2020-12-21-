import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000,32, 32,3)
print(y_train.shape)  #(50000,1)
print(x_test.shape) #(10000, 32, 32,3)
print(y_test.shape) #(10000,1)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(50000,32,96).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,32,96).astype('float32')/255. #이것도 가능 


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) #- 위에 /255 랑 똑같음 

from sklearn.preprocessing import OneHotEncoder
onehotencder = OneHotEncoder()
onehotencder.fit(y_train)
y_train = onehotencder.transform(y_train).toarray()
y_test = onehotencder.transform(y_test).toarray()

print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D

model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(32,96)))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Flatten())
# model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(10,activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=100, callbacks=[early_stopping], validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('loss, acc : ', loss, acc)

# loss, acc :  1.2108765840530396 0.5724999904632568

# loss, acc :  1.3426063060760498 0.5562999844551086 -Conv1D