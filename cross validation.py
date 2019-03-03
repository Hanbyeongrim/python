# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:06:53 2018

@author: A
"""

##기본
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_blobs(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)

print("test score : {}".format(logreg.score(X_test,y_test)))

##교차검증
#k겹 교차검증
import mglearn
mglearn.plots.plot_cross_validation()

#iris data
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target)#기본값 k=3
print("cross_val score : {}".format(scores))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("cross_val score : {}".format(scores))
print("cross_val score : {:.2f}".format(scores.mean()))

#계층별 교차검증
print("iris label : \n{}".format(iris.target))#1/3:0, 1/3:1, 1/3:2 > k=3사용시 정확도 0가능
mglearn.plots.plot_stratified_cross_validation()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
print("cross_val score : \n{}".format(cross_val_score(
        logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3)
print("cross_val score : \n{}".format(cross_val_score(
        logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3, shuffle=True, random_state=0)#데이터 섞기
print("cross_val score : \n{}".format(cross_val_score(
        logreg, iris.data, iris.target, cv=kfold)))

##loocv :폴드 하나에 샘플 하나만 있는 k겹 교차검증. 하나의 표인트만 테스트 셋. 작은데이터에 유용
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("cross_val count : ", len(scores))
print("cross_val score : {:.2f}".format(scores.mean()))

##shuffle split 교차검증 : train a, test b, split k회 지정하여 검증. 미선택 데이터 발생 가능
#실수 : %비율, 정수 : 개수, 샘플링의 일종, 계층별 기능도 있음 : StratifiedShuffleSplit
mglearn.plots.plot_shuffle_split()
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(train_size=0.5, test_size=0.5, n_splits=10)

scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("cross_val score : \n{}".format(scores))

##group 교차검증
mglearn.plots.plot_group_kfold()
from sklearn.model_selection import GroupKFold
X, y = make_blobs(n_samples=12, random_state=0)
groups = [0,0,0,1,1,1,1,2,2,3,3,3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("cross_val score : {}".format(scores))

