# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


#데이터 불러오기
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))

print(format(iris_dataset['data'][:5]))

#데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],iris_dataset['target'],random_state=0)

print(format(X_train.shape))

#산점도
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10,10),marker='o',
                           hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
## figsize : 그래프 크기 / marker : 점 모양 / s : 점 크기 / alpha : 점 투명도
## hist_kwds : 빈도그래프 모양, 구간

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print(format(X_new.shape))

prediction = knn.predict(X_new)
print("예측 : {}".format(prediction))
print("타겟 이름 : {}".format(iris_dataset['target_names'][prediction]))

#knn 평가
y_pred = knn.predict(X_test)
print("예측값 : {}".format(y_pred))
print("정확도 : {:.2f}".format(np.mean(y_pred == y_test)))