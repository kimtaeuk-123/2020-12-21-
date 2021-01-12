import numpy as np
import tensorflow as tf

x_train = np.load('../data/npy/mnist_x_train.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28,1)/255. #이것도 가능 
# (x_test.reshape(x_test)[0],x_test.shape[0],x_test.shape[1],x_test.shape[2],1) #나중엔 이렇게 

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
#OneHotencoding
#직접하기

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

#실습 완성하기
#지표는 acc /// 0.985 이상


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=100, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss, acc : ', loss, acc)