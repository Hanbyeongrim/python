# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:45:27 2018

@author: A
"""

##랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import mglearn

#moon data
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)#에스티 : 만들 트리 갯수
forest.fit(X_train, y_train)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,3,figsize=(10,10)) #2행 3열
for i, (ax, tree) in enumerate(zip(axes.ravel(),forest.estimators_)):#이누머 : 0부터 번호부여
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1,-1], alpha=0.4)
axes[-1,-1].set_title("random forest")
mglearn.discrete_scatter(X[:,0],X[:,1],y)

#breast cancer data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("train score : {:.3f}".format(forest.score(X_train, y_train)))
print("test score : {:.3f}".format(forest.score(X_test, y_test)))

#특성 중요도 그래프 정의하기
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(forest)

##그래디언트 부스팅 회귀 트리
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("train score : {:.3f}".format(gbrt.score(X_train, y_train)))#훈련 정확도 100이라 과적합
print("test score : {:.3f}".format(gbrt.score(X_test, y_test)))

#과적합 방지
#학습률 조정
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("train score : {:.3f}".format(gbrt.score(X_train, y_train)))
print("test score : {:.3f}".format(gbrt.score(X_test, y_test)))

#깊이 조정
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("train score : {:.3f}".format(gbrt.score(X_train, y_train)))
print("test score : {:.3f}".format(gbrt.score(X_test, y_test)))

plot_feature_importances_cancer(gbrt)