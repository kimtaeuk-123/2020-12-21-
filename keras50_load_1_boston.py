import numpy as np
import tensorflow as tf

x_data = np.load('../data/npy/boston_x.npy')
y_data = np.load('../data/npy/boston_y.npy')

# npy불러와서 파일 완성

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x_data, y_data, train_size=0.8, random_state=66, shuffle=True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x1_train)
x_train = scaler.transform(x1_train)
x_test = scaler.transform(x1_test)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1,1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1,1)
# y = y.reshape(506,1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=30, kernel_size=(2,1), padding='same', strides=1, input_shape=(13,1,1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100,activation= 'relu'))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')


model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x1_train, y1_train, epochs=100, validation_split=0.2, batch_size=8, callbacks=[early_stopping])

loss, mae = model.evaluate(x1_test, y1_test, batch_size=8)
print('loss, mae = ', loss, mae)

y1_predict = model.predict(x1_test)

from sklearn.metrics import r2_score

r2 = r2_score(y1_test,y1_predict)