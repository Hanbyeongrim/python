# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:56:47 2018

@author: A
"""

##estimation
##ARI,NMI : 군집결과와 비교할 대상 있을 때
import mglearn
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,4,figsize=(15,3),subplot_kw={"xticks":(),"yticks":()})
#알고리즘 리스트만들기
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
algorithms = [KMeans(n_clusters=2),AgglomerativeClustering(n_clusters=2),DBSCAN()]

#무작위로 클러스터 할당
import numpy as np
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters, cmap=mglearn.cm3,
    s=60, edgecolors="black")
axes[0].set_title("random assign - ARI : {:.2f}".format(
        adjusted_rand_score(y,random_clusters)))

for ax, algorithm in zip(axes[1:],algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,
               cmap=mglearn.cm3,s=60,edgecolors="black")
    ax.set_title("{} - ARI : {:.2f}".format(algorithm.__class__.__name__,
                 adjusted_rand_score(y, clusters)))

#accuracy가 아니라 adjusted_rand_score나 normalized_mutual_info_score 사용해야 함
from sklearn.metrics import accuracy_score
clusters1 = [0,0,1,1,0]
clusters2 = [1,1,0,0,1]
print("accuracy : {:.2f}".format(accuracy_score(clusters1, clusters2)))
print("ARI : {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))

##실루엣 계수 : 비교대상 없는경우. 그러나 복잡할 수록 잘 안맞음
from sklearn.metrics.cluster import silhouette_score
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15,3), subplot_kw={"xticks":(),"yticks":()})
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters, cmap=mglearn.cm3,
    s=60, edgecolors="black")
axes[0].set_title("random assign - ARI : {:.2f}".format(
        silhouette_score(X_scaled,random_clusters)))

for ax, algorithm in zip(axes[1:],algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,
               cmap=mglearn.cm3,s=60,edgecolors="black")
    ax.set_title("{} - ARI : {:.2f}".format(algorithm.__class__.__name__,
                 silhouette_score(X_scaled, clusters)))
    
##얼굴 데이터에서 고유값 찾고 변환
#pca (수동)
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten = True, random_state=0)

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):#타겟별 빈도가 달라 50개씩 일괄 추출
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255.

pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

#dbscan
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("eigen label : {}".format(np.unique(labels)))#결과 : -1. 모두 잡음이라는 의미

#min_sample 조정
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print("eigen label : {}".format(np.unique(labels)))

#eps 조정
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("eigen label : {}".format(np.unique(labels)))

#label 별 개수
#bincount는 음수 불가라 +1 처리해 계산
print("label count : {}".format(np.bincount(labels+1))) #잡음 32개

noise = X_people[labels==-1]
fig, axes = plt.subplots(3, 9, subplot_kw = {"xticks":(),"yticks":()},figsize=(12,4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape),vmin=0, vmax=1)
    
#eps별 클러스터 구분
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("eps = {}".format(eps))
    dbscan = DBSCAN(min_samples=3, eps = eps)
    labels = dbscan.fit_predict(X_pca)
    print("label count : {}".format(len(np.unique(labels))))
    print("label size : {}".format(np.bincount(labels+1)))#eps 7이 특이함
    
#eps=7
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

for cluster in range(max(labels)+1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
                             subplot_kw = {"xticks":(),"yticks":()})
    for image, label, ax in zip(X_people[mask],y_people[mask],axes):
        ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)
        ax.set_title(people.target_names[label].split()[-1])

#kmeans(자동)
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("kmeans cluster size : {}".format(np.bincount(labels_km)))

#pca성분으로 kmeans했기 때문에 pca.inverse_transform 사용해야함
fig, axes = plt.subplots(2, 5, figsize=(12, 4), subplot_kw = {"xticks":(),"yticks":()})
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),vmin=0,vmax=1)
    
#예시
mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)

#계층 군집
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("cluster size : {}".format(np.bincount(labels_agg)))
#계층과 kmeans 비교
print("ARI : {}".format(adjusted_rand_score(labels_agg, labels_km)))#공통부분 거의 없음

#덴드로그램
from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward(X_pca)
plt.figure(figsize=(20,5))
#p : 군집 갯수, no_label : 라벨 표시 안함
dendrogram(linkage_array, p=7, truncate_mode="level", no_labels=True)
plt.xlabel("sample No")
plt.ylabel("cluster distance")
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [36,36], "--",c="k")

#10개 클러스터 그림
#각 행은 한 클러스터 소속이고, 왼쪽 숫자는 갯수.
n_clusters = 10
for cluster in range(n_clusters):
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 10, figsize=(15, 8),
                             subplot_kw = {"xticks":(),"yticks":()})
    axes[0].set_ylabel(np.sum(mask))
    for image, label, asdf, ax in zip(
            X_people[mask],y_people[mask],labels_agg[mask],axes):
        ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={"fontsize":9})

#클러스터를 10개에서 40개로 늘려봄
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print("cluster size : {}".format(np.bincount(labels_agg)))

n_clusters = 40
for cluster in [10,13,19,22,36]:#아무 클러스터 번호
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 10, figsize=(15, 8),
                             subplot_kw = {"xticks":(),"yticks":()})
    cluster_size = np.sum(mask)
    axes[0].set_ylabel("#{} : {}".format(cluster, cluster_size))
    for image, label, asdf, ax in zip(
            X_people[mask],y_people[mask],labels_agg[mask],axes):
        ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={"fontsize":9})
    for i in range(cluster_size, 15):
        axes[i].set_visible(False)
        
