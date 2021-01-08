# sklearn 데이터셋
# LSTM 으로 모델링
#Dense  와 성능비교 
#이진분류

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer

#1 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x.shape) # (569,30)
# print(y.shape) # (569,)  스칼라가 569개인 벡터 하나   -> 총 31개

x = x.reshape(569, 30, 1)

print(x[:5])  # -전처리되어있는 느낌 
print(y)      # - 0, 1 나온다 -> 분류 

#전처리 알아서 / minmax, train_test_split



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(Dense(10, input_dim=30, activation='relu'))
# model.add(Dense(1))

model.add(LSTM(100, input_shape=(30,1), activation='relu')) # relu 0에서 무한대 , liner -무한대 에서 무한대, 우리가 원하는건 0에서 1 -> sigmoid 사용  
model.add(Dense(80, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
 #히든이없는 레이어 -> 가능 , 성능 좋음 그러나 히든레이어 있는 딥러닝이 쪼금 더 좋음


# model.compile(loss='mse', optimizer='adam', metrics='acc') #acc - 에큐러시
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy') # 이진분류할땐 나중에 배울때까지 크로스 엔트로피 사용 ,[accuracy]
# 이진분류할땐 나중에 배울때까지 크로스 엔트로피 사용 ,[accuracy]
# model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, validation_split=0.2, batch_size=2, callbacks=[early_stopping])

loss = model.evaluate(x, y)
print('loss = ', loss)

# 분류 모델에서는 accuracy 를 sklearn에 있는 accuracy_score 사용해야한다 - 나중에 알려주심 
# 실습 1 acc.0.985 이상
# 실습 2 predict 출력 

# loss, acc =  [0.16156363487243652, 0.940246045589447]
# loss, acc =  [0.31653890013694763, 0.8927943706512451]

# y[0:5] = ? 0 or 1
# y_pred = model.predict(x[-5:-1])
# print(y_pred)
# print(y[-5:-1])

# 결과치 나오게 코딩

# loss =  [0.6657088994979858, 0.6274164915084839]-LSTM

# loss =  [0.662835419178009, 0.6274164915084839]-LSTM 30/100 early_stopping