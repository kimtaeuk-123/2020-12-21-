# 실습 19_1, 2, 3, 4, 5, EalryStopping까지
# 6개 파일완성
# 첫번째거 튜닝후에 그거에맞춰서 나머지5개 비교
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target



print(np.max(x), np.min(x))


# print(x[:5])
# print(y[:10])
# print(x.shape, y.shape) # (442, 10) (442, )

# print(np.max(x), np.min(y))
# print(dataset.feature_names) # 6 culmn 
# print(dataset.DESCR)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(np.max(x), np.min(x))
print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=66)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=2)

loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_predict, y_test)

print('r2 = ', r2)

#1
# loss =  5766.53173828125
# mae =  58.91021728515625
# r2 =  0.11808364558981621

#2
# loss =  3911.123291015625
# mae =  50.34771728515625
# r2 =  0.1645862295761722

#3
# loss =  4570.1181640625
# mae =  53.2329216003418
# r2 =  0.1779916713288645



