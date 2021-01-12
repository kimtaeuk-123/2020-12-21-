# 인공지능계의 hellow world라 불리는 40_mnist2!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28,1)/255. #이것도 가능 

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
#OneHotencoding
#직접하기

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit(y_train)
y_train = onehotencoder.transform(y_train).toarray()
y_test = onehotencoder.transform(y_test).toarray()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2)))
# model.add(Conv2D(10, (2,2)))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))

#실습 완성하기
#지표는 acc /// 0.985 이상


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' #. <-지금 study 폴더, 02d -정수형 , f= float 


early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

model_checkpoint = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=10, verbose=1,validation_split=0.2, batch_size=10, callbacks=[early_stopping, model_checkpoint])

result = model.evaluate(x_test, y_test, batch_size=10)
print('loss : ', result[0])
print('acc : ', result[1])


#시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #면적잡기, 판깔기
plt.subplot(2,1,1) # 2행 1열 중 첫번째
#이미지 2개 뽑겠다. -> (2,1) 즉 2행 1열짜리 하나 만들겠다?
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('cost loss')
# plt.title('손실비용') 한글 안됨 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #<- 범주


plt.subplot(2,1,2) #2행 2열 중 두번째 
# plt.plot(hist.history['acc'], marker='.', c='red', label='acc') #accuracy 면 accuracy로 써야한다 
plt.plot(hist.history['acc'], marker='.', c='red') #accuracy 면 accuracy로 써야한다 
# plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid() #모눈종이 격자 

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc']) #<- 범주
plt.show()

#과제 1.matplotlib 한글깨짐 처리할것
