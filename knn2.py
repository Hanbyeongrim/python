# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:24:22 2018

@author: A
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#임의의 데이터 forge 산점도
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["클래스 0","클래스 1"],loc=4)
plt.xlabel("첫번째 특성")
plt.ylabel("두번째 특성")
print(format(X.shape))

#임의의 데이터 wave 산점도
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("특성")
plt.ylabel("타겟")

#실제 유방암 데이터
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(format(cancer.keys())) ##keys:R에서 str
print(format(cancer.data.shape))
print(format(cancer.featrue_names))
print(format({n: v for n, v in zip(cancer.target_names, 
                                   np.bincount(cancer.target))})) ##클래스 별 갯수

#보스턴 집값
from sklearn.datasets import load_boston
boston = load_boston()

#knn
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)


from sklearn.model_selection import train_test_split ##train, test 분할
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

print("예측 : {}".format(clf.predict(X_test))) ##predict는 :.2f안됨
print("정확도 : {:.2f}".format(clf.score(X_test, y_test)))

#결정 경계
fig, axes = plt.subplots(1,3, figsize=(10,3))
for n_neighbors, ax in zip([1,3,9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y) ##생성,적합을 한번에
    mglearn.plots.plot_2d_classification(clf, X, fill = True, eps=0.5, ax=ax,
                                         alpha = 0.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("0")
    ax.set_ylabel("1")
axes[0].legend(loc=3)

#성능 평가
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = [] ##훈련 정확도
test_accuracy = [] ##예측 정확도

neighbors_settings = range(1, 11) ##1<=n<11

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="train accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("accuracy")
plt.xlabel("neighbors")
plt.legend()

#knn 회귀
mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print("테스트 예측 : {}".format(reg.predict(X_test)))
print("테스트 R^2 : {:.2f}".format(reg.score(X_test, y_test)))

#램덤 생성 데이터
fig, axes = plt.subplots(1,3,figsize=(10,4)) ## 빈 그래프
line = np.linspace(-3,3,1000).reshape(-1,1) ## -3~3까지 난수 1000개, ??reshape(-1,1)
for n_neighbors, ax in zip([1, 3, 9], axes): ##zip 각 항들의 원소를 순서대로 매칭함
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1),markersize=8)
    
    ax.set_title(
            "{} neigbor score : {:.2f} test score : {:.2f}".format(
                    n_neighbors, reg.score(X_train, y_train),
                    reg.score(X_test, y_test)))
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
axes[0].legend(["predict", "train data/tar", "test data/tar"], loc="best")

