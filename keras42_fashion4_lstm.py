import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape) #(60000,28, 28)
# print(y_train.shape)  #(60000,)
# print(x_test.shape) #(10000, 28, 28)
# print(y_test.shape) #(10000,)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

x_train = x_train.reshape(60000,28,28).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28)/255. #이것도 가능 

from sklearn.preprocessing import OneHotEncoder
onehotencder = OneHotEncoder()
onehotencder.fit(y_train)
y_train = onehotencder.transform(y_train).toarray()
y_test = onehotencder.transform(y_test).toarray()

print(y_train.shape) #(60000,10)
print(y_test.shape)  #(10000, 10)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM

model = Sequential()

model.add(LSTM(100, input_shape=(28,28), activation='relu'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
# model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(10))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=100, callbacks=[early_stopping], validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print('loss, acc : ', loss, acc)

#loss, acc :  0.0817188173532486 0.3215999901294708
# loss, acc :  0.018811112269759178 0.8779000043869019 /255 + (28,28)
# loss, acc :  0.020366638898849487 0.8661999702453613  (16, 49) 로 바꿧더니 비슷하다.