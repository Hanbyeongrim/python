# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:23:03 2018

@author: A
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#선형회귀
mglearn.plots.plot_linear_regression_wave()

##wave 데이터
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
print("coef : {}".format(lr.coef_))
print("intercept : {}".format(lr.intercept_))

print("train score : {}".format(lr.score(X_train,y_train)))
print("test score : {}".format(lr.score(X_test,y_test)))

##boston 확장 데이터
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("train score : {}".format(lr.score(X_train,y_train)))
print("test score : {}".format(lr.score(X_test,y_test)))

#릿지 회귀
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("train score : {}".format(ridge.score(X_train, y_train)))
print("test score : {}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train) ##알파값 클수록 변수 제한함.
print("train score : {}".format(ridge10.score(X_train, y_train)))
print("test score : {}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train) ##알파값 작으면 일반 회귀와 비슷
print("train score : {}".format(ridge01.score(X_train, y_train)))
print("test score : {}".format(ridge01.score(X_test, y_test)))

#각 릿지 모델 별 계수 분포
plt.plot(ridge10.coef_,'^', label = "ridge alpha = 10")
plt.plot(ridge.coef_,'s', label = "ridge alpha = 1")
plt.plot(ridge01.coef_,'v', label = "ridge alpha = 0.1")

plt.plot(lr.coef_,'o', label = "linear regression")
plt.xlabel("position n") ##n 번째 계수
plt.ylabel("size") ##계수 크기
plt.ylim(-25,25)
plt.legend()

#학습곡선
mglearn.plots.plot_ridge_n_samples()

#라쏘 회귀
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train,y_train)
print("train score : {}".format(lasso.score(X_train, y_train)))
print("test score : {}".format(lasso.score(X_test, y_test)))
print("feature count : {}".format(np.sum(lasso.coef_!=0)))

lasso001 = Lasso(alpha = 0.01, max_iter=100000).fit(X_train,y_train)##max : 반복수
print("train score : {}".format(lasso001.score(X_train, y_train)))
print("test score : {}".format(lasso001.score(X_test, y_test)))
print("feature count : {}".format(np.sum(lasso001.coef_!=0)))

lasso00001 = Lasso(alpha = 0.0001, max_iter=100000).fit(X_train,y_train)##max : 반복수
print("train score : {}".format(lasso00001.score(X_train, y_train)))
print("test score : {}".format(lasso00001.score(X_test, y_test)))
print("feature count : {}".format(np.sum(lasso00001.coef_!=0)))

#각 라쏘 모델별 계수 분포
plt.plot(lasso.coef_,'s', label = "lasso alpha = 1")
plt.plot(lasso001.coef_,'^', label = "lasso alpha = 0.01")
plt.plot(lasso00001.coef_,'v', label = "lasso alpha = 0.0001")

plt.plot(ridge01.coef_,'o', label = "ridge alpha=0.1")
plt.xlabel("position n") ##n 번째 계수
plt.ylabel("size") ##계수 크기
plt.ylim(-25,25)
plt.legend(ncol=2, loc=(0,1.05))