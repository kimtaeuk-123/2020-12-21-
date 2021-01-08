from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# Conv2d = converlution2D
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(10,10,1)))
model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
model.add(Dense(1))

model.summary()
