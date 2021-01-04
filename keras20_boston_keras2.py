# 2개
#1.EarlyStooping 적용 하지않은최고모델
#2.EarlyStooping 적용한 최고모델
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x = scaler.transform(x_train)


from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam', metrics='mae')
model.fit(x_test, y_test, epochs=100, validation_split=0.2, verbose=1)

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
