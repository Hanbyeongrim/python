# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:14:24 2018

@author: A
"""

##병합군집(계층적 군집)
#병합군집 예시
import mglearn
mglearn.plots.plot_agglomerative_algorithm()

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:,0],X[:,1],assignment)
import matplotlib.pyplot as plt
plt.legend(["cluster 0","cluster 1","cluster 2"])

mglearn.plots.plot_agglomerative()

#덴드로그램
from scipy.cluster.hierarchy import dendrogram, ward
X, y = make_blobs(random_state=0, n_samples=12)
linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25],"--",c="k")
ax.plot(bounds, [4, 4],"--",c="k")

ax.text(bounds[1],7.25,"two cluster",va="center",fontdict={"size":15})
ax.text(bounds[1],4,"three cluster",va="center",fontdict={"size":15})
plt.xlabel("sample num")
plt.ylabel("cluster distance")

##DBSCAN
#random data
from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("cluster label : \n{}".format(clusters))

mglearn.plots.plot_dbscan()

#moon data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters, cmap=mglearn.cm2,
            s=60,edgecolors="black")
plt.xlabel("feature 0")
plt.ylabel("feature 1")