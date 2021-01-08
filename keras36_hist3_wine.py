from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR)
# print(dataset.feature_names)
x = dataset.data
y = dataset.target

# print(y) # 0,1,2 다중분류
# print(x.shape) # (178,13)
# print(y.shape) # (178,)

# x = x.reshape(178,13,1)

#실습, dnn 완성 할것
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()

model.add(Dense(50, input_dim=13, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(1, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])
hist = model.fit(x, y, epochs=100, validation_split=0.2, callbacks=[early_stopping])

loss = model.evaluate(x, y)
print('loss = ', loss)

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] -전처리 전 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] - 전처리 후 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] - 전처리 후 + early_stopping ???

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 빈자리에 설명(범주?)
plt.show()