import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape) #(50000, 32 ,32, 3)
# print(x_test.shape) #(10000, 32 ,32, 3)
# print(y_train.shape) #(50000,1)
# print(y_test.shape) #(10000,1)

# y_train = y_train.reshape(y_train.shape[0],1)
# y_test = y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,32,32,3)/255. #이것도 가능 

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

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


# loss, acc :  1.2882293462753296 0.5631999969482422
# r2 =  0.35806936458672