import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

#2. 모델 

model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()  #모델만 사용하기에 데이터가 없어도 모델이 구성된다 

# 모델 저장 

model.save("./model/save_keras35.h5") #경로저장 .<-점 하나가 현재 폴더 h5 확장자 
model.save(".//model//save_keras35_1.h5")
model.save(".\model\save_keras35_2.h5")
model.save(".\\model\\save_keras35_3.h5") #아무거나 사용 가능 

# \n 줄바꿈 -> \\n 으로 바꾼다  