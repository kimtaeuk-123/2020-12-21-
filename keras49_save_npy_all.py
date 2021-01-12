# boston, diabets, cancer, iris, wine - sklearn 
# mnist, fastion, cifar10, cifar100 - keras


from sklearn.datasets import load_boston , load_diabetes, load_breast_cancer, load_iris, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
#1. boston
boston_datasets = load_boston()
boston_x = boston_datasets.data
boston_y = boston_datasets.target

np.save('../data/npy/boston_x.npy', arr=boston_x) #1. study 폴더 안에 / 2. boston_x 라는게 들어있는 / 3. arr인 / 4. boston_x.npy파일을 만든다. 
np.save('../data/npy/boston_y.npy', arr=boston_y)
#2~5 까지 save파일을 만드시오

#2. dibets

diabets_datasets = load_diabetes()
diabets_x = diabets_datasets.data
diabets_y = diabets_datasets.target

np.save('../data/npy/diabets_x.npy', arr=diabets_x)
np.save('../data/npy/diabets_y.npy', arr=diabets_y)

#3. cancer

cancer_datasets = load_breast_cancer()
cancer_x = cancer_datasets.data
cancer_y = cancer_datasets.target

np.save('../data/npy/cancer_x.npy', arr=cancer_x)
np.save('../data/npy/cancer_y.npy', arr=cancer_y)

#4. iris

iris_datasets = load_iris()
iris_x = iris_datasets.data
iris_y = iris_datasets.target

np.save('../data/npy/iris_x.npy', arr=iris_x)
np.save('../data/npy/iris_y.npy', arr=iris_y)

#5. wine

wine_datasets = load_wine()
wine_x = wine_datasets.data
wine_y = wine_datasets.target

np.save('../data/npy/wine_x.npy', arr=wine_x)
np.save('../data/npy/wine_y.npy', arr=wine_y)

#6. mnist
(m_x_train, m_y_train), (m_x_test, m_y_test) = mnist.load_data()  #m= mnist
np.save('../data/npy/mnist_x_train.npy', arr=m_x_train)
np.save('../data/npy/mnist_x_test.npy', arr=m_x_test)
np.save('../data/npy/mnist_y_train.npy', arr=m_y_train)
np.save('../data/npy/mnist_y_test.npy', arr=m_y_test)

#7. cifar10
(c10_x_train, c10_y_train), (c10_x_test, c10_y_test) = cifar10.load_data()  #c10= cifar10
np.save('../data/npy/cifar10_x_train.npy', arr=c10_x_train)
np.save('../data/npy/cifar10_x_test.npy', arr=c10_x_test)
np.save('../data/npy/cifar10_y_train.npy', arr=c10_y_train)
np.save('../data/npy/cifar10_y_test.npy', arr=c10_y_test)

#8. cifar1010
(c100_x_train, c100_y_train), (c100_x_test, c100_y_test) = cifar100.load_data()  #c100= cifar100
np.save('../data/npy/cifar100_x_train.npy', arr=c100_x_train)
np.save('../data/npy/cifar100_x_test.npy', arr=c100_x_test)
np.save('../data/npy/cifar100_y_train.npy', arr=c100_y_train)
np.save('../data/npy/cifar100_y_test.npy', arr=c100_y_test)

#9. fashion_mnist
(f_x_train, f_y_train), (f_x_test, f_y_test) = fashion_mnist.load_data()  #f= fashion_mnist
np.save('../data/npy/fashion_mnist_x_train.npy', arr=f_x_train)
np.save('../data/npy/fashion_mnist_x_test.npy', arr=f_x_test)
np.save('../data/npy/fashion_mnist_y_train.npy', arr=f_y_train)
np.save('../data/npy/fashion_mnist_y_test.npy', arr=f_y_test)