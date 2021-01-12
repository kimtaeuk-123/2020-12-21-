from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()

print(dataset)
# 1.숫자 데이터가 나온다. 중요한것 -dictionary(딕셔너리)  - key value
# // data 와 array 쌍. target과 array쌍 
# //target_names': array(['setosa', 'versicolor', 'virginica']< 열 이름
# filename 경로로 쭉들어가면 iris 데이터가 있음 -쉽게 데이터확인 가능 (150, 4) ['setosa', 'versicolor', 'virginica']

print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values()) #? 안됨
x_data = dataset.data
y_data = dataset.target
x_data = dataset['data'] # 키값 '' <-형태가 스트링이라 '' 빼주면 안된다.
y_data = dataset['target']

# print(x)
# print(y)
# print(dataset.frame) #none 
# print(dataset.target_names) #['setosa' 'versicolor' 'virginica']
# print(dataset["DESCR"])
# print(dataset["feature_names"])
# print(dataset.filename)

print(type(x_data), type(y_data))
np.save('../data/npy/iris_x.npy', arr=x_data) 
np.save('../data/npy/iris_y.npy', arr=y_data) #arr= 이름 