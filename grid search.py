# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:59:03 2018

@author: A
"""

##그리드 서치 : 매개변수 조합 찾기
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("train data size : {} test data size : {}".format(X_train.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma = gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C' : C, 'gamma' : gamma}

print("best score : {:.2f}".format(best_score))
print("best parameters : {}".format(best_parameters))

#3분할 데이터의 필요
import mglearn
mglearn.plots.plot_threefold_split()

X_trainval, X_test, y_trainval, y_test = train_test_split(
        iris.data, iris.target, random_state=0)#훈련+검증, 테스트 분할

X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, random_state=1)#훈련, 검증 분할

print("train size : {}, valid size : {}, test size : {}".format(
        X_train.shape[0],X_valid.shape[0],X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma = gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C' : C, 'gamma' : gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)#훈련+검증데이터로 모델 적합
test_score = svm.score(X_test, y_test)
print("valid best score : {:.2f}".format(best_score))
print("best parameters : {}".format(best_parameters))
print("best score at best parameters : {:.2f}".format(test_score))#92%만 정확하게 분류

##교차검증 그리드 서치
mglearn.plots.plot_cross_val_selection()

import numpy as np
from sklearn.model_selection import cross_val_score

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma = gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C' : C, 'gamma' : gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
print("best score : {:.2f}".format(best_score))
print("best parameters : {}".format(best_parameters))

mglearn.plots.plot_grid_search_overview()

##gridsearchcv 이용
#딕셔너리 지정 필수
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(),param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

grid_search.fit(X_train, y_train)

print("test score : {:.2f}".format(grid_search.score(X_test, y_test)))
print("best param : {}".format(grid_search.best_params_))
print("best score : {:.2f}".format(grid_search.best_score_))
print("best model : \n{}".format(grid_search.best_estimator_))

#교차검증 그리드 시각화
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

scores = np.array(results.mean_test_score).reshape(6,6)

#c, gamma 히트맵 : 정확도 높으면 밝은색
mglearn.tools.heatmap(scores, xlabel="gamma", xticklabels=param_grid["gamma"],
                      ylabel="C", yticklabels=param_grid["C"], cmap="viridis")

#검색범위 부적절로 잘못 그려진 그래프
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13,5))
param_grid_linear = {"C":np.linspace(1,2,6), "gamma":np.linspace(1,2,6)}
param_grid_one_log = {"C":np.linspace(1,2,6), "gamma":np.logspace(-3,2,6)}
param_grid_range = {"C":np.logspace(-3,2,6), "gamma":np.logspace(-7,-2,6)}

for param_grid, ax in zip([param_grid_linear, 
                           param_grid_one_log, param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6,6)
    
    scores_image = mglearn.tools.heatmap(scores, xlabel="gamma", 
                   xticklabels=param_grid["gamma"], ylabel="C", 
                   yticklabels=param_grid["C"], cmap="viridis", ax=ax)
plt.colorbar(scores_image, ax=axes.tolist())

##비대칭 그리드 서치
#kernel조건도 들어있는 딕셔너리 생성
param_grid = [{'kernel' : ['rbf'],
              'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel' : ['linear'],
              'C' : [0.001, 0.01, 0.1, 1, 10, 100]}]
print("grid search list : \n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("best param : {}".format(grid_search.best_params_))
print("best score : {:.2f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
print(results.T)

##중첩 교차검증 : 모델 평가용
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma' : [0.001, 0.01, 0.1, 1, 10, 100]}
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                         iris.data, iris.target, cv=5)
print("score : ",scores)
print("mean score : ",scores.mean())#교차검증 정확도 평균이 98%임.

#안쪽 바깥쪽 다른 전략
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    import numpy as np
    outer_scores = []
    for training_samples, test_samples in outer_cv.split(X,y):
        best_params = {}
        best_score = -np.inf
        for parameters in parameter_grid:
            cv_scores = []
            for inner_train, inner_test in inner_cv.split(
                    X[training_samples], y[training_samples]):
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)#안쪽 교차검증 평균 점수
            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)

from sklearn.model_selection import ParameterGrid, StratifiedKFold
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),StratifiedKFold(5),
                   SVC, ParameterGrid(param_grid))
print("cross valid score : {}".format(scores))