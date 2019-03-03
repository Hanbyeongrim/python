# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:20:38 2018

@author: A
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

mglearn.plots.plot_animal_tree()

##tree
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#최대적합
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)#여기 오류로 이하 코드들 작동 x
print("train score : {:.3f}".format(tree.score(X_train, y_train)))
print("test score : {:.3f}".format(tree.score(X_test, y_test)))
#깊이 4로 제한
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("train score : {:.3f}".format(tree.score(X_train, y_train)))
print("test score : {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["bad","good"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

#특성 중요도
print("importance : \n{}".format(tree.feature_importances_))
#특성 중요도 그래프 정의하기
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)


tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

##ram price data로 트리와 회귀의 예측 비교
import os.path
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

plt.semilogy(ram_prices.date, ram_prices.price)

from sklearn.tree import DecisionTreeRegressor
data_train = ram_prices[ram_prices.date<2000]
data_test = ram_prices[ram_prices.date>=2000]

X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

#트리는 예측력 없음을 보여주는 그래프
plt.semilogy(data_train.date, data_train.price, label="train data")
plt.semilogy(data_test.date, data_test.price, label="test data")
plt.semilogy(ram_prices.date, price_tree, label="tree pred")
plt.semilogy(ram_prices.date, price_lr, label="linear pred")
plt.legend()