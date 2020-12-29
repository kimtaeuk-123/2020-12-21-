# validation_data 를 만들것
# train_test_split를 사용할것 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 data
x = np.array(range(1,101))
y = np.array(range(1,101))

'''x_train = x[:60] # 순서가 0번째부터 59번째 까지 ;;;; 값 1~60
x_val = x[60:80] # 61~80
x_test = x[80:]  # 81~100 
#리스트의 슬라이싱

y_train = y[:60] # 순서가 0번째부터 59번째 까지 ;;;; 값 1~60
y_val = y[60:80] # 61~80
y_test = y[80:]  # 81~100 
#list slising'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True) # Shuffle = Flase 는 똑바로 
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  
                                                  train_size=0.8, shuffle=True) 
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

'''#2. model 
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(4))

#model.add(Dense(1))
#3 compile

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=()) #우리가 필요한 최적의 웨이트값을 구하기 위해 엑스트레인 와이트레인을 넣는다 

#4. evaluate
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
print(y_predict)

#shuffle = False
#loss :  0.0027641593478620052
#mae :  0.04408388212323189

#shuffle = True
#loss :  0.0019195610657334328
#mae :  0.036717139184474945

#validdation = 0.2
#loss :  0.003187121357768774
#mae :  0.04608169198036194'''