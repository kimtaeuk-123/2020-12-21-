# 과제 및 실습 LSTM
# 전처리 얼리스탑핑 등등 다 넣을것!!
# 데이터 1~100 / 5개씩 잘라라
#     x             y
# 1,2,3,4,5         6
#...
#95,96,97,98,99    100

#predict 
#96,97,98,99,100 ->101
#...
#100,101,102,103,104 ->105

#예상 predict는 (101,102,103,104,105)

import numpy as np
import tensorflow as tf

a = np.array(range(1, 101)) 
size = 6


def split_x(seq,size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i :(i+size)]
        aaa.append ([item for item in subset])
        
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:, 0:-1] 
y = dataset[:, -1] 

print(x.shape) #(95, 5)
print(y.shape) #(95)

x = x.reshape(95, 5, 1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(5,1))
dense1 = LSTM(10, activation='relu')(input1)
dense2 = Dense(10)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3) 

model=Model(inputs=input1, outputs=output1)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# y = scaler.transform(y)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, validation_split=0.2, verbose=1, callbacks=[early_stopping])

b = np.array(range(96, 106))
size=6

x_predict = split_x(b, size)
# print(x_predict) #(96~105)
# print(x_predict.shape) #(5.6)

x_predict = x_predict.reshape(5,6, 1)
y_predict = model.predict(x_predict)
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss = ', loss)
print('y_predict = ', y_predict)

# y_predict =  [[123.850876]  전처리 전 
#  [125.01066 ]
#  [126.17176 ]
#  [127.33419 ]
#  [128.49794 ]]

# y_predict =  [[222.85931] 418/1000 early_stopping
#  [225.47067]
#  [228.0749 ]
#  [230.67212]
#  [233.26268]]

# y_predict =  [[107.13058] 1000/1000 early_stopping + train_test_split
#  [108.25031]
#  [109.37193]
#  [110.49538]
#  [111.6206 ]]