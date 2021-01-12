import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape) #(60000, 28 ,28)
# print(x_test.shape) #(10000, 28 ,28)
# print(y_train.shape) #(60000,)
# print(y_test.shape) #(10000,)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28,1)/255. #이것도 가능 

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit(y_train)
y_train = onehotencoder.transform(y_train).toarray()
y_test = onehotencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=100, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_predict)

print('r2 = ', r2)

# loss, mae :  0.33110445737838745 0.032514918595552444
# r2 =  0.812274112188013

# loss, acc :  0.3408927321434021 0.8797000050544739
# r2 =  0.8052391243872398