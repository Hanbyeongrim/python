# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:36:33 2018

@author: A
"""
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

##SVM(SVC)
#svm용 데이터
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8)
y = y % 2
#산점도
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
#직선 분류기 그린 산점도
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
#특성1^2한 특성 추가해서 3차원 산점도
X_new = np.hstack([X,X[:,1:]**2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()

ax = Axes3D(figure, elev=-152, azim=-26) #elev:상하방향, azim:좌우방향
mask = y == 0 #mask는 y가 0인가 아닌가를 저장
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask,2],c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask,2],c='r',marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1^2")
#3차원 산점도에 선형경계 그리기
#선형식 만들기
linear_svm_3d = LinearSVC().fit(X_new,y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
#그리기
figure = plt.figure()

ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:,0].min() - 2, X_new[:,0].max() + 2, 50)#a~b까지 n개 난수
yy = np.linspace(X_new[:,1].min() - 2, X_new[:,1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0]*XX+coef[1]*YY+intercept)/-coef[2]
ax.plot_surface(XX,YY,ZZ, rstride=8,cstride=8,alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask,2],c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask,2],c='r',marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1^2")

#평면에 그리기
ZZ = YY**2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape),levels=[dec.min(), 0, dec.max()],
            cmap=mglearn.cm2,alpha=0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

#본격적인 svm
from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps = 0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel()>0
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels, s=15,markeredgewidth=3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

#c와 gamma에 따른 분류모델 변화
fig, axes = plt.subplots(3, 3, figsize=(15,10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0,0].legend(["class0","class1","class0's sv","class1's sv"],ncol=4,loc=(0.9,1.2))

#가우시안 커널
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("train score : {:.2f}".format(svc.score(X_train,y_train)))
print("test score : {:.2f}".format(svc.score(X_test,y_test)))

plt.boxplot(X_train, manage_xticks=False)
plt.yscale("symlog")
plt.xlabel("features")
plt.ylabel("feature size")

#svm용 전처리 직접하기
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)

X_train_scaled = (X_train - min_on_training) / range_on_training #최대최소 표준화>0~1
print("min : {}".format(X_train_scaled.min(axis=0)))
print("max : {}".format(X_train_scaled.max(axis=0)))

X_test_scaled=(X_test - min_on_training)/range_on_training
svc = SVC()#c, gamma 디폴트
svc.fit(X_train_scaled, y_train)
print("train score : {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("test score : {:.3f}".format(svc.score(X_test_scaled,y_test)))

svc = SVC(gamma=0.1,C=1000)
svc.fit(X_train_scaled, y_train)
print("train score : {:.3f}".format(svc.score(X_train_scaled,y_train)))
print("test score : {:.3f}".format(svc.score(X_test_scaled,y_test)))