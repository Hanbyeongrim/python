# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:24:21 2018

@author: A
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#활성함수 렐루, 하이퍼탄젠트
line = np.linspace(-3,3,100)
plt.plot(line, np.tanh(line),label="tanh")
plt.plot(line, np.maximum(line,0),label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu,tanh")

##신경망
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100,noise=0.25,random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#기본 mlp(은닉층 : 1, 은닉노드 : 100)
mlp = MLPClassifier(solver='lbfgs',random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

##은닉노드 10
#히든레이어사이즈 각 인자가 층, 숫자가 해당 층 노드 수 /엑티베이션 : 활성함수, 렐루가 디폴트
mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10])
mlp.fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

##은닉층 : 2, 은닉노드 10, 10, 활성함수 : tanh
mlp = MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10],activation="tanh")
mlp.fit(X_train,y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

##알파값으로 복잡도 조정
fig, axes = plt.subplots(2,4,figsize=(20,8))
for axx, n_hidden_nodes in zip(axes,[10,100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs',random_state=0,
                            hidden_layer_sizes=[n_hidden_nodes,n_hidden_nodes],
                            alpha=alpha)
        mlp.fit(X_train,y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
        mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
        ax.set_title("n_hidden=[{},{}]\nalpha={:.4f}".format(
                n_hidden_nodes,n_hidden_nodes,alpha))
        
##초기값 조정
fig, axes = plt.subplots(2,4,figsize=(20,8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs',random_state=i,
                            hidden_layer_sizes=[100,100])
    mlp.fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,ax=ax)
    
##유방암 데이터
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train,y_train)

print("train score : {:.2f}".format(mlp.score(X_train,y_train)))
print("test score : {:.2f}".format(mlp.score(X_test,y_test)))

##표준화하기
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
#반복횟수때문에 결과는 나오지만 오류 발생
mlp=MLPClassifier(random_state=0).fit(X_train_scaled,y_train)

print("train score : {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("test score : {:.3f}".format(mlp.score(X_test_scaled,y_test)))

#반복횟수 1000
mlp=MLPClassifier(max_iter=1000, random_state=0).fit(X_train_scaled,y_train)

print("train score : {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("test score : {:.3f}".format(mlp.score(X_test_scaled,y_test)))

#반복횟수 1000, alpha=1
mlp=MLPClassifier(max_iter=1000, alpha=1, random_state=0).fit(X_train_scaled,y_train)

print("train score : {:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("test score : {:.3f}".format(mlp.score(X_test_scaled,y_test)))