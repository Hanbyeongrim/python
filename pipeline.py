# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:52:12 2018

@author: A
"""

##파이프 라인
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

svm = SVC()
svm.fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)
print("test score : {:.3f}".format(svm.score(X_test_scaled, y_test)))

#매개변수 찾기
from sklearn.model_selection import GridSearchCV
param_grid = {"C" : [0.001, 0.01, 0.1, 1, 10, 100],
              "gamma" : [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("best cv score : {:.3f}".format(grid.best_score_))
print("test score : {:.3f}".format(grid.score(X_test_scaled, y_test)))
print("best param : ",grid.best_params_)

#위의 코드드은 모든 데이터를 변환과 그리드 서치에 사용해버림(정보누설). 낙관적 결과 생성
import mglearn
mglearn.plots.plot_improper_processing()
#교차 검증의 분할은 모든 전처리과정에 앞서서, 모든 처리는 훈련부분에만 적용(교차검증 안에 있어야함)

##파이프라인 구축
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
#scaler라는 이름의 민맥스스케일러, svm이름의 SVC모델
pipe.fit(X_train, y_train)
print("test score : {:.3f}".format(pipe.score(X_test, y_test)))

#그리드 서치 이용
param_grid = {"svm__C" : [0.001, 0.01, 0.1, 1, 10, 100],
              "svm__gamma" : [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print("best cv score : {:.3f}".format(grid.best_score_))
print("test score : {:.3f}".format(grid.score(X_test, y_test)))
print("best param : ",grid.best_params_)

mglearn.plots.plot_proper_processing()

#파이프라인을 이용한 예측기
def fit(self, X, y):
   X_transformed = X
   for name, estimator in self.steps[:-1]:
       X_transformed = estimator.fit_transform(X_transformed, y)#fit, transform 반복
   self.steps[-1][1].fit(X_transformed, y)#fit만
   return self
def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        X_transformed = step[1].transform(X_transformed)
    return self.steps[-1][1].predict(X_transformed)


#####정보누설
#무작위 데이터 생성
import numpy as np
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100,10000))
y = rnd.normal(size=(100,))

from sklearn.feature_selection import SelectPercentile, f_regression
select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print("X_selected shape : {}".format(X_selected.shape))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
print("cross val score(릿지) : {:.3f}".format(np.mean(
        cross_val_score(Ridge(), X_selected, y, cv=5))))
#무작위 데이터라 연관이 없을텐데 R^2값이 0.91로 좋게 나옴 > 전체 데이터 사용 때문
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,percentile=5)), 
                 ("ridge", Ridge())])
print("cross val score : {:.3f}".format(np.mean(cross_val_score(pipe, X, y, cv=5))))
#파이프라인 사용시 R^2값 음수 > 정보누설 막음

##make_pipeline : 단계 이름을 자동으로 생성
from sklearn.pipeline import make_pipeline
pipe_long = Pipeline([("scaler",MinMaxScaler()),("svm",SVC(C=100))])#보통
pipe_short = make_pipeline(MinMaxScaler(),SVC(C=100))#간소화
print("pipeline step : \n{}".format(pipe_short.steps))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pipe = make_pipeline(StandardScaler(),PCA(n_components=2), StandardScaler())
print("pipeline step : \n{}".format(pipe.steps))#같은게 반복되면 숫자도부여

#파이프라인 단계 속성확인
pipe.fit(cancer.data)
components = pipe.named_steps["pca"].components_
print("components.shape : {}".format(components.shape))

#그리드서치에서 파이프라인 속성확인
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C' : [0.01, 0.1, 1, 10, 100]}

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("best model : \n{}".format(grid.best_estimator_))
print("logreg step : \n{}".format(
        grid.best_estimator_.named_steps["logisticregression"]))#전체 모델중 로지스틱만
print("logreg estimator : \n{}".format(
        grid.best_estimator_.named_steps["logisticregression"].coef_))#로지스틱 계수

#그리드서치 매개변수에 따른 평균 점수
from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {"polynomialfeatures__degree" : [1,2,3], 
              "ridge__alpha" : [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

mglearn.tools.heatmap(grid.cv_results_["mean_test_score"].reshape(3, -1),
                      xlabel="ridge__alpha",ylabel="polynomialfeatures__degree",
                      xticklabels=param_grid["ridge__alpha"],
                      yticklabels=param_grid["polynomialfeatures__degree"],vmin=0)
print("best param : {}".format(grid.best_params_))
print("test score : {:.2f}".format(grid.score(X_test, y_test)))

param_grid = {"ridge__alpha" : [0.001, 0.01, 0.1, 1, 10, 100]}#다항식 특성 제거
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("score without poly : {:.2f}".format(grid.score(X_test, y_test)))

#모델선택
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
param_grid = [{"classifier":[SVC()],"preprocessing":[StandardScaler()],
                             "classifier__gamma":[0.001,0.01,0.1,1,10,100],
                             "classifier__C":[0.001,0.01,0.1,1,10,100]},
    {"classifier":[RandomForestClassifier(n_estimators=100)],"preprocessing":[None],
                             "classifier__max_features":[1,2,3]}]

X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("best param : \n{}\n".format(grid.best_params_))#svc, c=10, gamma=0.01, 전처리:스탠드
print("best cross val score : {:.2f}".format(grid.best_score_))
print("test score : {:.2f}".format(grid.score(X_test, y_test)))