# []:[] mlp 함수형
# keras10_mlp3.py를 카피 함수형

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1,101), range(201, 301)])

x = np.transpose(x)
y = np.transpose(y)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66)

input1 = Input(shape=(3,)) #함수형 모델, 시퀀셜 모델이랑 성능 동일 
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(3)(dense3)
model = Model(inputs = input1, outputs = outputs)

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_split=0.2, verbose=1)

loss, mae = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):                              #mse만있고 rmse가 없어서 루트를 씌워줘야 한다. , 원래 있던 y_test 에 새로운 y_predict 를 비교
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt -  루트
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


