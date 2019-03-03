# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:03:56 2018

@author: A
"""

##k means cluster
#k평균 군집 예시
import mglearn
mglearn.plots.plot_kmeans_algorithm()
mglearn.plots.plot_kmeans_boundaries()

##random data
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state = 1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("cluster label : \n{}".format(kmeans.labels_))
print(kmeans.predict(X))

mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,markers="o")
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
                         [0,1,2],markers="^",markeredgewidth=2)

#군집수 바꿔보기
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2,figsize=(10,5))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0],X[:,1],assignments,ax=axes[0])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0],X[:,1],assignments,ax=axes[1])

##군집분석 실패 케이스
#밀도가 다른 데이터
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1,2.5,0.5],random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

mglearn.discrete_scatter(X_varied[:,0],X_varied[:,1],y_pred)
plt.legend(["cluster0","cluster1","cluster2"],loc="best")#cluster0,1은 먼 것도 포함
plt.xlabel("feature 0")
plt.ylabel("feature 1")

#원형이 아닌 데이터
import numpy as np
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2,2))#데이터를 늘어지도록 변환
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

mglearn.discrete_scatter(X[:,0],X[:,1],kmeans.labels_,markers="o")
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
                         [0,1,2],markers="^",markeredgewidth=2)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

#two moon data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=2).fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred, cmap=mglearn.cm2, s=60, edgecolors="k")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker="^",
            c=[mglearn.cm2(0),mglearn.cm2(1)],s=100, linewidth=2, edgecolors="k")
plt.xlabel("feature 0")
plt.ylabel("feature 1")

##벡터 양자화(k평균 군집의 중점을 분해로 볼 수 있다는 취지)
#face data
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):#타겟별 빈도가 달라 50개씩 일괄 추출
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, random_state=0)

from sklearn.decomposition import NMF
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)

from sklearn.decomposition import PCA
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test),nmf.components_)

fig, axes = plt.subplots(3,5,figsize=(8,8),subplot_kw={"xticks":(),"yticks":()})
fig.suptitle("extracted feature")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_,
                                               pca.components_,nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap="viridis")
    ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0,0].set_ylabel("kmeans")
axes[1,0].set_ylabel("pca")
axes[2,0].set_ylabel("nmf")

fig, axes = plt.subplots(4,5,figsize=(8,8),subplot_kw={"xticks":(),"yticks":()})
fig.suptitle("reconstructed feature")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T,X_test,X_reconstructed_kmeans,
                                               X_reconstructed_pca,X_reconstructed_nmf):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape), cmap="viridis")
    ax[3].imshow(rec_nmf.reshape(image_shape))
axes[0,0].set_ylabel("original")
axes[1,0].set_ylabel("kmeans")
axes[2,0].set_ylabel("pca")
axes[3,0].set_ylabel("nmf")

#moon data
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred, cmap="Paired", s=60, edgecolors="black")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker="^",
            c=range(kmeans.n_clusters),s=60, linewidth=2, edgecolors="black",cmap="Paird")
plt.xlabel("feature 0")
plt.ylabel("feature 1")
print("cluster label : \n{}".format(y_pred))

distance_features = kmeans.transform(X)
print("distance shape : {}".format(distance_features.shape))
print("distance of cluster : {}".format(distance_features))