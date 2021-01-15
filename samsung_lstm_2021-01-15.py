import numpy as np
import tensorflow as tf
import csv
import pandas as pd
import datetime

df = pd.read_csv('./test/samsung.csv', index_col=0, header=0,engine='python', encoding='CP949', thousands=',')
dfdf = pd.read_csv('./test/samsung2.csv',index_col=0, header=0, engine='python', encoding='CP949', thousands=',')

df.drop(df.columns[[4]], axis=1, inplace=True) # 등락률 삭제 
dfdf.drop(dfdf.columns[[4,5,6]], axis=1, inplace=True)#전일비, ? ,등락율 삭제 

df = df.sort_values(by=['일자'], axis=0) #samsung1 데이터 일자 순으로 
dfdf = dfdf.sort_values(by=['일자'], axis=0) #samsung2 데이터 일자 순으로 

df = df.iloc[:,0:5]  # ~ 2021-01-13) 시가 , 고가 ,저가 , 종가, 거래량 
dfdf = dfdf.iloc[1:2,0:5] # 2021-01-14 시가 , 고가 ,저가 , 종가, 거래량 

df_merge = pd.concat([df, dfdf]) #전체일자 

df_merge = df_merge.iloc[1739:,0:5].astype('float32') # 2018-05-04 부터 액면분할 , (662,5)


x = df_merge.iloc[:661, 0:5].astype('float32')   #(661,5)
y = df_merge.iloc[1:662, 3].astype('float32')    #(661,) 
x_predict = df_merge.iloc[661, 0:5] #(5,)
x_predict = x_predict.values.reshape(1,-1)  #(1,5)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

print(x_train.shape) # (528,5)
print(x_test.shape) # (133,5)
print(y_train.shape) # (528,)
print(y_test.shape) # (132,)

print(x_train)

x_train = x_train.reshape(528, 5, 1)
x_test = x_test.reshape(133, 5, 1)
x_predict = x_predict.reshape(1,5,1) # (1,5)--> (1,5,1)

np.save('../data/npy/samsung2.npy', arr=[x_train, y_train, x_test, y_test, x_predict])


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten


model = Sequential()
model.add(LSTM(500, input_shape=(5,1),activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(11))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint

early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')

modelpath = '../data/modelcheckpoint/samsung_{epoch:02d}-{val_loss:08f}.hdf5'
modelcheckpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=2, callbacks=[early_stopping,modelcheckpoint], validation_split=0.3)

loss, acc = model.evaluate(x_test, y_test, batch_size=2)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_predict)

print('y_predict =', y_predict)

model = load_model('../data/modelcheckpoint/samsung_25-6627912.000000.hdf5') #좋은 웨이트값을 k52_1_mnist_checkpoint.hdf5로파일명을 고쳐야한다
y_predict = model.predict(x_predict)
print(y_predict)

model.save_weights('../data/h5/samsung_weight.h5')