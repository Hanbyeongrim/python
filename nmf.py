# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:59:10 2018

@author: A
"""

##비음수행렬분해(NMF)
#nmf 예시
import mglearn
mglearn.plots.plot_nmf_illustration()

##face data
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

import numpy as np
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):#타겟별 빈도가 달라 50개씩 일괄 추출
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, random_state=0)
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)

from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3,5,figsize=(15,12),
                         subplot_kw={"xticks":(),"yticks":()})
for i, (component, ax) in enumerate(zip(nmf.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("comp {}".format(i))
 
#4번째 성분으로 각 사진 정렬(원형)
compn = 3
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig, axes = plt.subplots(2,5,figsize=(15,8),
                         subplot_kw={"xticks":(),"yticks":()})
for i, (inds, ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[inds].reshape(image_shape))

#8번째 성분으로 각 사진 정렬(왼쪽 방향)
compn = 7
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig, axes = plt.subplots(2,5,figsize=(15,8),
                         subplot_kw={"xticks":(),"yticks":()})
for i, (inds, ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[inds].reshape(image_shape))
    
##3개 신호가 합성된 데이터
S = mglearn.datasets.make_signals()
plt.figure(figsize=(6,1))
plt.plot(S, "-")
plt.xlabel("time")
plt.ylabel("signal")

A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S, A.T)
print("data shape : {}".format(X.shape))

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)#데이터 복원
print("recover data shape : {}".format(S_.shape))

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X,S,S_,H]
names = ["measured data","3 composed data","nmf recover data","pca recover data"]
fig, axes = plt.subplots(4, figsize=(8,4), gridspec_kw={"hspace":0.5},
                         subplot_kw={"xticks":(),"yticks":()})
for model, names, ax in zip(models, names, axes):
    ax.set_title(names)
    ax.plot(model[:,:3],"-")
