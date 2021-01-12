import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
from tensorflow.keras.datasets import cifar100

(x_train, x_test), (y_train, y_test) = cifar100.load_data()

print(x_train.shape) #(60000, 28, 28)
print(x_test.shape) #(60000,)
print(y_train.shape) #(10000, 28, 28)
print(y_test.shape) #(10000,)

plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
plt.show()   #0~255 있는데 0일수록 검은색, 255일수록 밝은색 

