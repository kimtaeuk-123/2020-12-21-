import numpy as np
# 실습 다:1 앙상블을 구현하시오.

#1.데이터 구성
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101,201), range(411,511),range(100,200)])
y1 = np.array([range(711, 811), range(1,101), range(201, 301)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
from sklearn.model_selection import train_test_split
x2_train, x2_test = train_test_split(x2, train_size=0.8, shuffle=False)
# from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, = train_test_split(x1, x2, y1, shuffle=False, train_size=0.8)
#2. 모델 구성

from tensorflow.keras.models import Sequential, Model  #Model = 함수형
from tensorflow.keras.layers import Dense, Input

#3. 모델 1

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
#output1 = Dense(3,)(dense1)

#3. 모델 2

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(3,)(dense2)


# 모델 concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate

merge1= concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

#모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

#모델 선언
model = Model(inputs=[input1, input2], outputs=output1)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=1, validation_split=0.2, verbose=1)

loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print('loss = ', loss)

print('model.metrics_names : ', model.metrics_names ) # dense_11_loss : 첫번째 summary에서 임의로 지정. dense_15_loss: 두번째

y1_predict = model.predict([x1_test, x2_test])


'''
print("============================")
print("y1_predict : \n", y1_predict)
print("============================")
print("y2_predict : \n", y2_predict)
print("============================")
'''
from sklearn.metrics import mean_squared_error
def RMSE(y_predict, y_test):
    return np.sqrt(mean_squared_error(y_predict, y_test))
print("RMSE : ", RMSE(y1_test, y1_predict))

# RMSE1 = RMSE(y1_test, y1_predict)

# print("RMSE1 : ", RMSE1)

# from sklearn.metrics import mean_squared_error
# def RMSE2(y2_predict, y2_test):
#     return np.sqrt(mean_squared_error(y2_predict, y2_test))
# print("RMSE : ", RMSE2)



from sklearn.metrics import r2_score
r2 = r2_score(y1_predict,y1_test)

print("R2 : ", r2)


# 훈련 해보면 metrics mse랑 mae랑 다름 1<-대표로스 (1.loss+2.loss, 1.metrics+2.metrics), 2(1.loss), 3(2.loss), 4(1.metrics), 5(2.metrics)