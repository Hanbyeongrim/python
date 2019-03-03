# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:55:48 2018

@author: A
"""

##전처리 예시 p169,170
import mglearn
mglearn.plots.plot_scaling()

##cancer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=1)

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
print("after scale shape :{}".format(X_train_scaled.shape))
print("befor min : \n{}".format(X_train.min(axis=0)))
print("befor max : \n{}".format(X_train.max(axis=0)))
print("after min : \n{}".format(X_train_scaled.min(axis=0)))
print("after max : \n{}".format(X_train_scaled.max(axis=0)))

#트랜스 폼 했으나 1을 넘어가는 경우 생김. 트레인의 최대최소를 이용하기 때문.
X_test_scaled = scaler.transform(X_test)
print("after min : \n{}".format(X_test_scaled.min(axis=0)))
print("after max : \n{}".format(X_test_scaled.max(axis=0)))

##트랜스 폼에서 최대, 최소 사용시
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=5, test_size=0.1)

fig, axes = plt.subplots(1, 3, figsize=(13,4))
axes[0].scatter(X_train[:,0],X_train[:,1],c=mglearn.cm2(0), label="train set",s=60)
axes[0].scatter(X_test[:,0],X_test[:,1],c=mglearn.cm2(1), 
    label="test set",s=60,marker="^")
axes[0].legend(loc="upper left")
axes[0].set_title("original data")

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0), 
    label="train set",s=60)
axes[1].scatter(X_test_scaled[:,0],X_test_scaled[:,1],c=mglearn.cm2(1), 
    label="test set",s=60,marker="^")
axes[1].set_title("transform data")

#예시를 위한 방법일 뿐 사용하면 안되는 방법(트레인을 먼저하는 것이 아니라 테스트를 따로하는 방법)
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:,0],X_train_scaled[:,1],c=mglearn.cm2(0), 
    label="train set",s=60)
axes[2].scatter(X_test_scaled_badly[:,0],X_test_scaled_badly[:,1],c=mglearn.cm2(1), 
    label="test set",s=60,marker="^")
axes[2].set_title("badly transform data")

for ax in axes:
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    
##cancer SVC
from sklearn.svm import SVC
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
svm=SVC(C=100)
svm.fit(X_train, y_train)
print("accuracy : {:.2f}".format(svm.score(X_test,y_test)))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("minmax scale accuracy : {:.2f}".format(svm.score(X_test_scaled, y_test)))

#standart scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("standard scale accuracy : {:.2f}".format(svm.score(X_test_scaled, y_test)))

#robust scale
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("robust scale accuracy : {:.2f}".format(svm.score(X_test_scaled, y_test)))

#normalizer scale
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("standard scale accuracy : {:.2f}".format(svm.score(X_test_scaled, y_test)))