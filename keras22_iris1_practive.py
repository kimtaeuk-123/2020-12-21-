import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris

# x, y= load_iris(retrun_X_y=True) # 이것도 있다.



dataset = load_iris()

x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x.shape) #(150, 4)
print(y.shape) #(150, )
print(x[:5])
print(y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=4, activation='sigmoid'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(x, y, batch_size=8)
print('loss, acc', loss, acc)

# from sklearn.metrics import r2_score
# r2= r2_score(x, y)
# print('r2 = ', r2_score)
