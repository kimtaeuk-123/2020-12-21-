import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import load_iris

dataset = load_iris()

# x = dataset.data
# y = dataset.target
x= dataset['data'] 
y= dataset['target'] #여러가지 형태가 있다. 헤매지말것 

# print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

# print(dataset.values())

# print(dataset.target_names) #names< s 주의
# array(['setosa', 'versicolor', 'virginica']

print(x.shape) # (150,4)
print(y.shape) # (150,)
print(type(x), type(y)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

df = pd.DataFrame(x, columns=dataset['feature_names'])
# df = pd.DataFrame(x, columns=dataset.feature_names)
# print(df)
# print(df.shape)
# print(df.columns)
# print(df.index)

# print(df.head()) #위에서 부터 5개 #print(df[:5])
# print(df.tail()) #아래에서 부터 5개 #print(df[:-5])
# print(df.info()) #Non-Null
# print(df.describe()) 

df.columns = ['sepal_length','sepal_width','petal_length','petal_width'] # cloumn 명을 바꾼다. (cm 사라짐)
print(df.describe()) 
print(df.info())

# y칼럼을 추가해 보아요
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head()) # Target 추가됨

print(df.shape) #(150,5) <- (150,4)에서 1 늘어남
# print(df.columns)
# print(df.index)
# print(df.tail()) 

# print(df.info())
# print(df.isnull()) #False - 넓값이 없다
# print(df.isnull().sum())
# print(df.describe())
# print(df['Target'].value_counts()) #2가    50개 1이 50개 0이 50개]

# 상관관계, 히트맵
# print(df.corr()) #sepal_length-sepal_length 1.000000 100% 상관관계 

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale =1.0)
# sns.heatmap(data=df.corr(), square=True, annot =True, cbar=True)  # squre - 사각형 annot-수치 표시 cbar - 옆에 색에따라 달라지는거 표시
# plt.show()

# 도수 분포도
plt.figure(figsize=(10,6))

plt.subplot(2,2,1) #2행 2열 을 그릴건데 1번째
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x='petal_length', data=df)
plt.title('petal_legth')

plt.subplot(2, 2, 4)
plt.hist(x='petal_width', data=df)
plt.title('petal_width')

plt.show()
