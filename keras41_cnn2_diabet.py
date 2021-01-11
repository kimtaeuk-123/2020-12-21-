
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle=True, random_state=66)
print(x_train.shape) #(282, 10)
print(x_test.shape) #(89, 10)
print(y_train.shape) #(282,)
print(y_test.shape) #(89,)


from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
minmaxscaler.fit(x_train)
x_train = minmaxscaler.transform(x_train)
x_test = minmaxscaler.transform(x_test)
x_val = minmaxscaler.transform(x_val)


# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder()
# onehotencoder.fit(y_train)
# y_train = onehotencoder.transform(y_train).toarray()
# y_test = onehotencoder.transform(y_test).toarray()
x_train = x_train.reshape(282, 10, 1, 1)
x_test = x_test.reshape(89, 10, 1, 1)
x_val = x_val.reshape(71,10,1,1)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape) #(71,10)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(filters=160, kernel_size=(2,1), padding='same', strides=1, input_shape=(10,1,1) ))
model.add(Dense(100))
model.add(Dense(100))
# model.add(Conv2D(160, (2,1)))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Flatten())
model.add(Dense(1))


model.compile(loss = 'mse', optimizer='adam', metrics='mae')

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=1000, validation_data=(x_val,y_val), batch_size=10, callbacks=[earlystopping], verbose=1)

loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)



from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict) #... (y_predict, y_test) 랑 값이 다름!!

print('r2 = ', r2)

# loss =  3265.189697265625
# mae =  47.71310043334961
# r2 =  0.49689219950333285

#acc 값이 안나옴 

# loss =  3222.420654296875
# mae =  47.202701568603516
# r2 =  0.5034821949382067 to_categorical






