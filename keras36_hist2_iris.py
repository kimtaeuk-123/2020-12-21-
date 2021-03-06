import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris

# x, y= load_iris(retrun_X_y=True) # 이것도 있다.



dataset = load_iris()

x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x.shape) #(150, 4)
print(y.shape) #(150, )
print(x[:5])
print(y)

from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical 위에랑 동일

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)
print(y_train.shape)#(120,3)  #print(y.shape) (150,3)
print(y_test)
print(y_test.shape) #(30,3)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(x_test, y_test) 
print('loss, acc', loss, acc)

# from sklearn.metrics import r2_score
# r2= r2_score(x, y)
# print('r2 = ', r2_score)

# y_pred = model.predict(x[-5:-1])
# print(y_pred)
# print(y[-5:-1])

# 결과치 나오게 수정argmax


# np.argmax(y_pred,axis=1)
# print(y_pred[0].argmax())

# print(y.argmax())

# loss, 0.6322639584541321

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 빈자리에 설명(범주?)
plt.show()