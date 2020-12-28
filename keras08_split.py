from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np
from numpy import array
# np.array()
# -> array()

#1. 데이터
x = np.array(range(1, 101))
#x = np.array(range(100)) 0~99
y = np.array(range(101, 201))

x_train = x[:60] # 순서가 0번째부터 59번째 까지 ;;;; 값 1~60
x_val = x[60:80] # 61~80
x_test = x[80:]  # 81~100 
#리스트의 슬라이싱

y_train = y[:60] # 순서가 0번째부터 59번째 까지 ;;;; 값 1~60
y_val = y[60:80] # 61~80
y_test = y[80:]  # 81~100 


x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15,])
y_test = array([11,12,13,14,15,])
x_pred = array([16,17,18])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) #8개는 트레이닝, 2개는 평가 ??
#model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) #8개는 트레이닝, 2개는 평가 ??

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, msa : ", results)

y_predict = model.predict(x_test)
#print("y_predict : ", y_predict)

#np.sqrt(resluts[0]) 음..??

# 사이킷런
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):                              #mse만있고 rmse가 없어서 루트를 씌워줘야 한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt -  루트
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test) )

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
