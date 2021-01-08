# 인공지능계의 hellow world라 불리는 mnist!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(x_train[0])
print(x_train[0].shape) #(28,28)

plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
plt.show()   #0~255 있는데 0일수록 검은색, 255일수록 밝은색 
