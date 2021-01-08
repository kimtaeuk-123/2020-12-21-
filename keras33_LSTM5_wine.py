# sklearn 데이터셋
# LSTM 으로 모델링
#Dense  와 성능비교 
# 다중분류

from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR)
# print(dataset.feature_names)
x = dataset.data
y = dataset.target

# print(y) # 0,1,2 다중분류
# print(x.shape) # (178,13)
# print(y.shape) # (178,)

x = x.reshape(178,13,1)

#실습, dnn 완성 할것
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(50, input_shape=(13,1), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(1, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])
model.fit(x, y, epochs=100, validation_split=0.2)

loss = model.evaluate(x, y)
print('loss = ', loss)

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] -전처리 전 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] - 전처리 후 

# loss =  [0.0, 0.3988763988018036, 0.601123571395874] -LSTM  ????