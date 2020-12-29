import numpy as np

#1.data

#x = np.array([range(100), range(301, 401), range(1, 101)])
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301,401)])
y = np.array([range(711, 811), range(1,101)])
print(x.shape)   #(5, 100)
print(y.shape)   #(2, 100)
x_pred2 = np.array([100, 402, 101, 100, 401])
print("x_pred2.shape : ", x_pred2.shape) #(5,)
x = np.transpose(x)
y = np.transpose(y)
#x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5)

print(x.shape) #(100,5)
print(y.shape) #(100,2)
print("x_pred2.shape :", x_pred2.shape)# (1,5) <- 바뀌지않기 때문에 행렬이 아니다 

#z = x.reshape(10,2)
print(x.shape) #(3,100)
print(y.shape) #(3,100)# shape 권장함. (10,) = 스칼라가 10개  -> (2, 10) 2행 10열  행 무시 열 우선  (11, 3 이면 input_dim= 3)
#print(z.shape)
'''
x = np.transpose(x)
y = np.transpose(y)
print(x)
print(x.shape) # (100,3)
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66) # 순서 기억, 랜덤 스테이트:랜덤 단수 고정 

print(x_train.shape) #(80, 5)
print(y_train.shape) #(80, 2)

#2. model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense 2.0 때부터 위에 것보다 안좋음  
model = Sequential()
# model.add(Dense(10, input_dim=5))
model.add(Dense(10, input_shape=(5,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) # y 의 열  # 하이퍼 파라미터 튜닝

#3 compile

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 50, batch_size=1, validation_split=0.2, verbose=1) # 16, 5 16, 2  verbose 훈련되는과정을 안보여줌. 장점: 시간단축

'''
verbose=0 : 안나옴
verbose=1 : 다 출력  = 기본값(디폴트) 
verbose=2 : ============== 출력 안함 
verbose=3 : 훈련만 반복하는게 나옴 
'''

 #20% 검증데이터. 엑스:[1,2], [11,12] 그리고 와이:[1,2] 씀

#4 평가 예측

loss, mae = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)
#print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):                              #mse만있고 rmse가 없어서 루트를 씌워줘야 한다. , 원래 있던 y_test 에 새로운 y_predict 를 비교
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt -  루트
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)