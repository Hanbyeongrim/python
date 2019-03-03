# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:12:37 2018

@author: A
"""

##PCA
#pca 예시
import mglearn
mglearn.plots.plot_pca_illustration()

#cancer data
#histogram
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins, color=mglearn.cm3(0),alpha=0.5)
    ax[i].hist(benign[:,i],bins=bins, color=mglearn.cm3(2),alpha=0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("feature size")
ax[0].set_ylabel("frequency")
ax[0].legend(["bad","good"],loc="best")
fig.tight_layout()

#pca
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)#주요한 주성분 갯수 한정
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("original data : {}".format(str(X_scaled.shape)))
print("pca data : {}".format(str(X_pca.shape)))

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(["bad","good"],loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("first comp")
plt.ylabel("second comp")

print("comp : {}".format(pca.components_))

#hitmap
plt.matshow(pca.components_, cmap="viridis")
plt.yticks([0,1],["first comp","second comp"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=60,ha="left")
plt.xlabel("feature")
plt.ylabel("comp")

##face data
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={"xticks":(),"yticks":()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
    
print("peopel.images.shape : {}".format(people.images.shape))
print("number of class : {}".format(len(people.target_names)))

#타겟별 빈도
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25}{1:3},".format(name, count), end=" ")
    if (i + 1) % 3 == 0:
        print()
        
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):#타겟별 빈도가 달라 50개씩 일괄 추출
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255. #0~255사이의 값이라 스케일의 일종으로 볼 수 있음

#knn으로 훈련
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("1-knn score : {}".format(knn.score(X_test, y_test)))

#화이트닝 옵션 : pca + standard scale
mglearn.plots.plot_pca_whitening()
pca = PCA(n_components=100, whiten = True, random_state =0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape : {}".format(X_train_pca.shape))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("test score : {}".format(knn.score(X_test_pca, y_test)))

print("pca.components_.shape : {}".format(pca.components_.shape))
fig, axes = plt.subplots(3, 5, figsize=(15,12),
                         subplot_kw={"xticks":(),"yticks":()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),cmap="viridis")
    ax.set_title("comp : {}".format(i+1))
    
#example that comp invert to original
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

#주성분 두개만으로 구분해 보기 -> 구별하기 어려움
mglearn.discrete_scatter(X_train_pca[:,0],X_train_pca[:,1],y_train)
plt.xlabel("first comp")
plt.ylabel("second comp")