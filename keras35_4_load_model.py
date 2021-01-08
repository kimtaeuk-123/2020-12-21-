import numpy as np
a = np.array(range(1, 11))

from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')
# 요 밑 3줄 넣고 테스트 
from tensorflow.keras.layers import Dense
model.add(Dense(5, name='kingkeras1')) # 레이어 이름 : dense 
model.add(Dense(1, name='kingkeras2')) # 레이어 이름 : dense_1 
###################

model.summary()

# 모델 저장



