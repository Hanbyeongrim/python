# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:49:00 2018

@author: A
"""

#임의의 데이터, 그래디언트부스팅분류 생성, 2class
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles

X, y = make_circles(noise=0.25, factor=0.5,random_state=1)

y_named = np.array(["blue","red"])[y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

##decision function
print("X_test.shape : {}".format(X_test.shape))
print("decision_function shape : {}".format(gbrt.decision_function(X_test).shape))
#아래 값이 양수면 1 음수면 0
print("decision_function : \n{}".format(gbrt.decision_function(X_test)))

print("compare : \n{}".format(gbrt.decision_function(X_test)>0))
print("predict:\n{}".format(gbrt.predict(X_test)))

greater_zero = (gbrt.decision_function(X_test)>0).astype(int)#t,f를 1,0으로 변환
pred = gbrt.classes_[greater_zero]
print("compare pred, predict :{}".format(np.all(pred==gbrt.predict(X_test))))

decision_function = gbrt.decision_function(X_test)
print("decision_function min : {:.2f}, max : {:.2f}".format(np.min(decision_function),
      np.max(decision_function)))#최대값,최소값의 의미를 파악하기는 어려움

#결정경계, 결정함수
import matplotlib.pyplot as plt
import mglearn
fig, axes = plt.subplots(1, 2, figsize=(13,5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0],alpha=0.4,
                                fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=0.4, cm=mglearn.ReBl)

for ax in axes:
    mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,
                             markers="^",ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,
                             markers="^",ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["test class0","test class1","train class0","train class0"],
    ncol=4, loc=(0.1,1.1))

##predict_proba
print("shape of predict_proba : {}".format(gbrt.predict_proba(X_test).shape))

print("probability : \n{}".format(gbrt.predict_proba(X_test)))

#결정경계, 결정함수
fig, axes = plt.subplots(1, 2, figsize=(13,5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0],alpha=0.4,
                                fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=0.5, cm=mglearn.ReBl,
                                            function="predict_proba")

for ax in axes:
    mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,
                             markers="^",ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,
                             markers="^",ax=ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["test class0","test class1","train class0","train class0"],
    ncol=4, loc=(0.1,1.1))

##iris data 3 class
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

#decision_function
print("dicision_function shape : {}".format(gbrt.decision_function(X_test).shape))
print("dicision_funtion result : \n{}".format(gbrt.decision_function(X_test)))

print("result of dicision_function : \n{}".format(
        np.argmax(gbrt.decision_function(X_test),axis=1)))
print("predict : \n{}".format(gbrt.predict(X_test)))

#predict_proba
print("predict_proba shape : {}".format(gbrt.predict_proba(X_test)))
print("sum : \n{}".format(gbrt.predict_proba(X_test).sum(axis=1)))

print("result of predict_proba : \n{}".format(
        np.argmax(gbrt.predict_proba(X_test),axis=1)))
print("predict : \n{}".format(gbrt.predict(X_test)))

#클래스 이름 가져오기
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("train class : {}".format(logreg.classes_))
print("predict : {}".format(logreg.predict(X_test),axis=1))
argmax_dec_func = np.argmax(logreg.decision_function(X_test),axis=1)
print("index : {}".format(argmax_dec_func[:10]))
print("index classes connect : {}".format(logreg.classes_[argmax_dec_func][:10]))