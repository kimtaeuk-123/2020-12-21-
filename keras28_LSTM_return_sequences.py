# keras23_3 copy
# LSTM 2개를 만들것

#model.add(LSTM(10, input_shape=(3,1)))
#model.add(LSTM(10))
import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])
# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) #(13,3)
print(y.shape) #(13,)
print(x_pred.shape) #(3,)

x_pred=x_pred.reshape(1, 3, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape) # (10,3,1)
# print(x_test.shape) #(3, 3, 1)
# print(y_train.shape) #(10,)
# print(y_test.shape) #(3,)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(3,1),return_sequences=True, activation='relu'))
model.add(LSTM(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()


'''
model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=1)

loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_pred)


print('loss : ', loss)
print('y_pred', y_pred)

# loss :  [4.074560642242432, 0.0]
# y_pred [[72.37549]]
# 2개 이상 연결은 무조건 좋아지거나, 안좋아지는게 아니라 튜닝, 데이터에 따라 다르다.
'''