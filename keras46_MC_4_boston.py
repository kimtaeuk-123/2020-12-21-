#keras41_cnn1_boston copy

import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) #(506, 13)
print(y.shape) #(506,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x1_train)
x_train = scaler.transform(x1_train)
x_test = scaler.transform(x1_test)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1,1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1,1)
# y = y.reshape(506,1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=30, kernel_size=(2,1), padding='same', strides=1, input_shape=(13,1,1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Dense(100))
# model.add(Dropout(0.2))
model.add(Dense(100,activation= 'relu'))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')


model.compile(loss = 'mse', optimizer='adam', metrics='mae')
hist = model.fit(x1_train, y1_train, epochs=100, validation_split=0.2, batch_size=8, callbacks=[early_stopping])

loss, mae = model.evaluate(x1_test, y1_test, batch_size=8)
print('loss, mae = ', loss, mae)

y1_predict = model.predict(x1_test)

from sklearn.metrics import r2_score

r2 = r2_score(y1_test,y1_predict)

print('r2 = ', r2)

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
