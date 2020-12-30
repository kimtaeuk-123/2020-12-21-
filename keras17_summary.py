import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])  #데이터 셋 = 행
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential #폴더구조 , 마지막만 쓸꺼니까 import 씀
from tensorflow.keras.layers import Dense #기초적

model = Sequential()
model.add(Dense(200, input_dim=1, activation='linear')) # input dimention 한계??
model.add(Dense(300, activation='linear'))
model.add(Dense(400))
model.add(Dense(600))
model.add(Dense(1))
model.summary() #none - 행 무시
# 파라미터 사진 확인 2x5= 10, 6x3 = 18, 4x4= 16, 5x1=5

# 실습2 + 과제
#ensemble1, 2, 3, 4 에 대해 서머리를 계산하고
#이해한 것을과제로 제출할 것 
#layer를 만들때 'name 이란놈에 대해 확인하고 설명할 것 
# 얘를 반드시 써야할 때가 있다. 그때를 말하라.