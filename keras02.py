# 네이밍 룰 = import 사용하면 길게 쓰지 않아도 된다.
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras import models
#from tensorflow import keras 
#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델구성
model = Sequential()
#model = models.Sequential()
#model = keras.models.Sequential
model.add(Dense(5, input_dim=1, activation='relu')) # 리니어보다 렐루가 상황에 따라 더 좋다 
model.add(Dense(3))  # 위에꺼 처럼 안쓴 이유는 디폴트(기본값)가 있어서  
model.add(Dense(4))  
model.add(Dense(1))

#3. 컴파일, 훈련


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result = model.predict([9])
print("result : ", result)

#실습 epochs 100 으로 9의 근사값을 도출하시오
