import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) #(442, 10)

x = x.reshape(442, 10, 1)/255.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(10,1)))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(80))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=2, callbacks=[early_stopping])

loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('r2 = ', r2)

# loss =  5766.53173828125
# mae =  58.91021728515625
# r2 =  0.11808364558981621

# loss =  3690.6904296875  -lstm
# mae =  47.666378021240234
# r2 =  -0.5280655788514825


# loss =  3533.245361328125 -lstm early stopping 200/200
# mae =  48.06595993041992
# r2 =  0.09881674389540629



