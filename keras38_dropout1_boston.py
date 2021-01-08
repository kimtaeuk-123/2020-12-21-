#EalryStopping
import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

print("===============================")
print(x[:5]) #0~4
print(y[:10]) 

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names)
# print(dataset.DESCR) #describe 

#data (MinMax)

# x = x /711.  #0 ~ 1 
# x = (x - 최소) (최대 -최소 )
# x = (x - np.min(x)) / (np.max(x)- np.min(x))
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.max(x), np.min(x))
# print(np.max(x[0]))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  
                                                  train_size=0.2, shuffle=True) 
scaler.fit(x_val)
x_val = scaler.transform(x_val)




from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(13,))
# model.add(Dropout(0.2))  # 128* 0.2 = 25개 사용 안함
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dropout(0.2))  # 128* 0.2 = 25개 사용 안함
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # 64* 0.2 = 12개 사용 안함
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # 128* 0.2 = 25개 사용 안함
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # 128* 0.2 = 25개 사용 안함
model.add(Dense(64, activation='relu'))
model.add(Dense(1))




# input1 = Input(shape=(13,))
# dense1 = Dense(100, activation='relu')(input1)
# dense1 = Dense(100)(dense1)
# dense1 = Dense(10)(dense1)
# output1 = Dense(1)(dense1)



# model = Model(input1, output1)

model.compile(loss = 'mse', optimizer='adam', metrics='mae')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping =EarlyStopping(monitor='loss', patience=20, mode='auto') # loss down 10qjs ckasmsek , auto-min,max

model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), batch_size=8, callbacks=[early_stopping])


#4 evaluate

loss, mae = model.evaluate(x_test, y_test, batch_size=8, )
print('loss, mae = ', loss, mae)

y1_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y1_predict, y_test)


print('r2 = ', r2)

# loss, mae =  16.858667373657227 2.9712021350860596
# r2 =  0.7938000536056695 전처리 전

# loss, mae =  11.793758392333984 2.6175220012664795
# r2 =  0.847403366879222 전처리 후 [X/711.]

# loss, mae =  9.03050422668457 2.062678337097168
# r2 =  0.8975892517708454 전처리 후 x MinMaxScalar

# loss, mae =  11.111374855041504 2.192822217941284
# r2 =  0.8902375321910196 제대로 전처리

# loss, mae =  6.755013942718506 1.9877015352249146
# r2 =  0.926742723768882 x_val 전처리 y는 못함

# loss, mae =  10.60881519317627 2.4592068195343018
# r2 =  0.8987011551646943 early stopping

# loss, mae =  16.516820907592773 3.0362768173217773
# r2 =  0.6993888435697218 # Dropout 

# loss, mae =  9.127985954284668 2.252638101577759
# r2 =  0.8981829912451086 # Dropout 2번째

