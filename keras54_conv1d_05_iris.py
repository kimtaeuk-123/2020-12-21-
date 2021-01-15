import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
x = x.reshape(150,4,1)/255.


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical 위에랑 동일

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train)
print(y_train.shape)#(120,3)  #print(y.shape) (150,3)
print(y_test)
print(y_test.shape) #(30,3)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(4,1), activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100,activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

loss= model.evaluate(x_test, y_test) 
print('loss : ', loss)


y_pred = model.predict(x[-5:-1])



# np.argmax(y_pred,axis=1)
# print(y_pred[0].argmax())

# print(y.argmax())

# loss :  [0.15085560083389282, 1.0] - LSTM

# loss :  [0.6759670376777649, 0.7333333492279053] - LSTM , early_stopping

# loss :  [0.26070261001586914, 0.8666666746139526] - conv1D