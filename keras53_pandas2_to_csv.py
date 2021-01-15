import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import load_iris

dataset = load_iris()

# x = dataset.data
# y = dataset.target
x= dataset['data'] 
y= dataset['target'] #여러가지 형태가 있다. 헤매지말것 

print(x.shape) # (150,4)
print(y.shape) # (150,)

df = pd.DataFrame(x, columns=dataset['feature_names'])


df.columns = ['sepal_length','sepal_width','petal_length','petal_width'] # cloumn 명을 바꾼다. (cm 사라짐)

# y칼럼을 추가해 보아요
print(df['sepal_length'])
df['Target'] = dataset.target

print(df.shape) # (150,5)

df.to_csv('../data/csv/iris_sklearn.csv',sep='.') # . 로 구분하겠다.