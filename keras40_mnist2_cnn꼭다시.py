# 인공지능계의 hellow world라 불리는 mnist!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# print(x_train[0])
# print(x_train[0].shape) #(28,28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()   #0~255 있는데 0일수록 검은색, 255일수록 밝은색 


x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28,1)/255. #이것도 가능 
# (x_test.reshape(x_test)[0],x_test.shape[0],x_test.shape[1],x_test.shape[2],1) #나중엔 이렇게 

#OneHotencoding
#직접하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(1, activation='softmax'))

#실습 완성하기
#지표는 acc /// 0.985 이상


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=100, validation_split=0.5, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss : ', loss)

#응용
#y_test 10개와 y_test 10개를 출력하시오

# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)

# loss :  [2.0908985137939453, 0.14980000257492065] - 튜닝 전 
# loss :  [2.331256628036499, 0.16259999573230743]- earlystopping 
# loss :  [1.6868547201156616, 0.14509999752044678]- ephocs=10