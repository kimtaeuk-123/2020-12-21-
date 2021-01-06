import numpy as np
import tensorflow as tf

a = np.array(range(1, 11)) 
size = 5

def split_x(seq,size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i :(i+size)]
        aaa.append ([item for item in subset])
        
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)


x = dataset[:, 0:-1] #(6,4) : - 모든것 
y = dataset[:, -1] #(6,)

print(x)
print(y)

x = x.reshape(6,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=66)

from tensorflow.keras.callbacks import EarlyStopping
ealry_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')  

from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(4,1))
dense1 = LSTM(10, activation='relu')(input1)
dense2 = Dense(40)(dense1)
dense3 = Dense(40)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200,  batch_size=1, callbacks=[ealry_stopping])

loss = model.evaluate(x_test , y_test)
print('loss : ', loss)

# loss :  [1.752826452255249, 0.0]
