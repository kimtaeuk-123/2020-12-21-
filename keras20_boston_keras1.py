# 2개
#1.EarlyStooping 적용 하지않은최고모델
#2.EarlyStooping 적용한 최고모델

# from tensorflow.keras.datasets import boston_housing #control + space

#이걸로만들라
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
x_train = train_data
y_train = train_target
x_test = test_data
y_test = test_target

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, shuffle=False, random_state=1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_dim=13, activation='relu'))
model.add(Dense(100, ))
model.add(Dense(100, ))
model.add(Dense(100, ))
model.add(Dense(100, ))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode='min')


model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping],batch_size=1)

loss = model.evaluate(x_test, y_test,batch_size=1)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(y_predict, y_test))
print('RMSE : ', RMSE)

from sklearn.metrics import r2_score
r2 = r2_score(y_predict, y_test)
print('r2 : ', r2)
