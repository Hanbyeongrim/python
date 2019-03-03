# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:36:26 2018

@author: A
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

##이진분류
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2,figsize=(10,3))

for model, ax in zip([LinearSVC(), LogisticRegression()],axes):
    clf = model.fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=0.7)#fill 배경 채우기, eps 밀집도
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("fig 0")
    ax.set_ylabel("fig 1")
axes[0].legend()

##로지스틱
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train,y_train)
print("train score : {:.3f}".format(logreg.score(X_train, y_train)))
print("test score : {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train,y_train)
print("train score : {:.3f}".format(logreg100.score(X_train, y_train)))
print("test score : {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("train score : {:.3f}".format(logreg001.score(X_train, y_train)))
print("test score : {:.3f}".format(logreg001.score(X_test, y_test)))

#각 모델별 계수 분포
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.plot(logreg100.coef_.T,'^',label="C=100")
plt.plot(logreg001.coef_.T,'v',label="C=0.01")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)#그래프틀, 로테이션 변수명 기울기
plt.hlines(0,0,cancer.data.shape[1]) #수평축(높이, 시작점, 끝점)
plt.ylim(-5,5)
plt.legend()

##L1 규제
for C, marker in zip([0.001, 1, 100], ['o','^','v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("C={:.3f} logreg train score : {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("C={:.3f} logreg test score : {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)#그래프틀, 로테이션 변수명 기울기
plt.hlines(0,0,cancer.data.shape[1]) #수평축(높이, 시작점, 끝점)
plt.ylim(-5,5)
plt.legend(loc=3)

##다중분류
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["class0","class1","class2"])

linear_svm = LinearSVC().fit(X,y)
print("계수 배열 : ",linear_svm.coef_.shape)
print("절편 배열 : ",linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_,linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)

#예측결과
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept)/coef[1],c=color)