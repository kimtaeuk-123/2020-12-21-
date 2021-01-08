# 텐서플로 데이터셋
# LSTM 으로 모델링
#Dense  와 성능비교 
# 회귀모델

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import boston_housing

# dataset = boston_housing.load_data()
# x = dataset.data
# y = dataset.target

# print(x.shape)
# print(y.shape)



(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

# print(x_train.shape) # (404,13)
# print(x_test.shape) # (102,13)

x_train = x_train.reshape(404,13,1)
x_test = x_test.reshape(102,13,1)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x = scaler.transform(x_train)


from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input , LSTM

model = Sequential()
model.add(LSTM(100, input_shape=(13,1), activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='mse',optimizer='adam', metrics='mae')
model.fit(x_test, y_test, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

loss, mae = model.evaluate(x_test,y_test)
print('loss, mae = ', loss , mae )

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(y_predict,y_test))
print('RMSE = ', RMSE)

from sklearn.metrics import r2_score
r2 = r2_score(y_predict, y_test)
print('r2 = ', r2)


# model = Sequential()
# model.add(Dense(10, input_dim=))

#이걸로만들라


# loss, mae =  31.659648895263672 4.01893424987793 -LSTM
# RMSE =  5.626690977926493
# r2 =  0.39124669486769104

# loss, mae =  16.617027282714844 2.8702871799468994 -LSTM , early_stopping
# RMSE =  4.076398884674746
# r2 =  0.751249759584639