# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:47:18 2018

@author: A
"""

##t-sne
##write data
from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,5,figsize=(10,5),
                         subplot_kw={"xticks":(),"yticks":()})
for ax, img in zip(axes.ravel(),digits.images):
    ax.imshow(img)
    
#pca를 이용해 2차원으로 축소(주성분 2개)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(digits.data)

digits_pca = pca.transform(digits.data)
colors = ["#476A2A","#7851B8","#BD3430","#4A2D4E","#875525",
          "#A83683","#4E655E","#853541","#3A3120","#535D8E"]
plt.figure(figsize=(10,10))
plt.xlim(digits_pca[:,0].min(),digits_pca[:,0].max())
plt.ylim(digits_pca[:,1].min(),digits_pca[:,1].max())

for i in range(len(digits.data)):
    plt.text(digits_pca[i,0],digits_pca[i,1],str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={"weight":"bold","size":9})
plt.xlabel("first comp")
plt.ylabel("second comp")#0,4,6은 비교적 잘 분리 됨

#tsne로 축소
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:,0].min(),digits_tsne[:,0].max())
plt.ylim(digits_tsne[:,1].min(),digits_tsne[:,1].max())

for i in range(len(digits.data)):
    plt.text(digits_tsne[i,0],digits_tsne[i,1],str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={"weight":"bold","size":9})
plt.xlabel("first tsne feature")
plt.ylabel("second tsne feature")#1,9만 애매하고 웬만큼 잘 구분 됨