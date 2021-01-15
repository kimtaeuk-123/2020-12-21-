from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR)
# print(dataset.feature_names)
x = dataset.data
y = dataset.target

print(x.shape) #(178,13)
print(y.shape) #(178,)

x = x.reshape(178,13,1)/255.
y = y.reshape(178,1)
#실습, dnn 완성 할것
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder()
# onehotencoder.fit(y_train)
# y_train = onehotencoder.transform(y_train)
# y_test = onehotencoder.transform(y_test)

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y_train.shape) #(142,3)
print(y_test.shape) #(36,3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()

model.add(Conv1D(filters=10, kernel_size=2, input_shape=(13,1), activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Flatten())
model.add(Dense(3,activation= 'softmax'))

from tensorflow.keras.callbacks import EarlyStopping
ealry_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, callbacks=[ealry_stopping], batch_size=10, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss, acc = ', loss, acc)

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] -전처리 전 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] - 전처리 후 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] -LSTM  ????

# loss, acc =  0.18920806050300598 0.9444444179534912 - Conv1D