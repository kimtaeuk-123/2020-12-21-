#1. 데이터
import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)      시계열데이터는 y를 만들기전엔 없다.
y = np.array([4,5,6,7])                            #(4,)           <1개씩 잘라서 작업하겠다> -> 4, 3, *1* (2개도 가능) #행 열 몇개씩 자르는지.

x = x.reshape(4, 3, 1)

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM             #LSTM은 *3차원*을 받아들인다. , Dense 는 2차원 RNN 4차원 ? DNN=Dense

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,1)))   # input_shape는 한차원씩 줄어든다. 1개의 timestep이 3개있음 
model.add(Dense(20))                                        # 4 x ( 1 + 1 + 10) x 10 = 480
model.add(Dense(10))                                        # 4개의 게이트(sigmoid 3개, tanh(탄젠트)1개), input, 바이어스, output, 10번(output)
model.add(Dense(1))                                         # (4,3) 4x3 -> (4,3,1) 4x3x1 똑같다  (5,4) -> (5,4,1), (5,2,2) 데이터 맞춰줘야 한다.

model.summary()
#480


#컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)



#4. 평가 예측

loss = model.evaluate(x)

print('loss : ', loss)

x_pred = np.array([5,6,7]) #(3,) 행은하나 reshape후에 -> (1, 3, 1) 데이터 자체는 안바뀜. 단지 lstm의구조로 바뀜 
x_pred = x_pred.reshape(1,3,1)   # 와꾸 맞춰야한다.
# 8이 나와야 한다 

result = model.predict(x_pred)
print(result) #[8.259014]

#행 열 몇개씩 자르는지.


# LSTM > GRU > RNN  LSTM이 가장 복잡하고 성능이 좋음. LSTM확장이 GRU. RNN이 가장 간단하지만 데이터가 적으면 좋을듯  