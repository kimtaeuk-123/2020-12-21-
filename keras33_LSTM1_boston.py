# sklearn 데이터셋
# LSTM 으로 모델링
#Dense  와 성능비교 
#회귀모델

import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

x = x.reshape(506,13,1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

print(x1_train.shape)
print(x1_test.shape)
print(y1_train.shape)
print(y1_test.shape)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(13,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

# input1 = Input(shape=(13,))
# dense1 = Dense(100, activation='relu')(input1)
# dense1 = Dense(100)(dense1)
# dense1 = Dense(10)(dense1)
# output1 = Dense(1)(dense1)
# model = Model(input1, output1)

model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x1_train, y1_train, epochs=100, validation_split=0.2, batch_size=8, callbacks=[early_stopping])

loss, mae = model.evaluate(x1_test, y1_test, batch_size=8)
print('loss, mae = ', loss, mae)

y1_predict = model.predict(x1_test)



from sklearn.metrics import r2_score
r2 = r2_score(y1_predict, y1_test)


print('r2 = ', r2)

# loss, mae =  11.093852043151855 2.5004611015319824
# r2 =  0.8640866004312779
