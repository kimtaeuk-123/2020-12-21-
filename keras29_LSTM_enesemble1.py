import numpy as np
import tensorflow as tf

x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])

x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])


x1_predict = x1_predict.reshape(1,3)
x2_predict = x2_predict.reshape(1,3)
print(x1.shape) #(13,3)
print(x2.shape) #(13,3)
print(y.shape) #(13,)
print(x1_predict.shape) #(3,)
print(x2_predict.shape) #(3,)
# x1=x1.reshpae(x1.shape[0],x1.shape[1],1)
# x2=x2.reshpae(x2.shape[0],x1.shape[1],1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, train_size=0.8, shuffle=True, random_state=66)
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10)(dense2)

merge1 = concatenate([dense1, dense2])
middle1 = Dense(10, activation='relu')(merge1)
middle1 = Dense(10)(middle1)

output1 = Dense(10)(middle1)
output1 = Dense(1)(output1)





model = Model(inputs=[input1, input2], outputs=output1)

model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit([x1_train,x2_train], y_train, epochs=100)

loss = model.evaluate([x1_test,x2_test], y_test)


y1_predict = model.predict([x1_predict, x2_predict])

print('loss = ', loss)
print('y_predict', y1_predict)
