# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:52:49 2018

@author: A
"""

##특성 선택
##일변량 통계 선택(F통계)
#cancer data
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

import numpy as np #노이즈 생성해서 추가
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data),50))
X_w_noise = np.hstack([cancer.data,noise])

X_train, X_test, y_train, y_test = train_test_split(
        X_w_noise, cancer.target, random_state=0, test_size=0.5)

select = SelectPercentile(percentile=50)#특성의 50%만 선택
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print("X_train.shape : {}".format(X_train.shape))
print("X_train_selected.shape : {}".format(X_train_selected.shape))
mask = select.get_support()#선택된거 안된거 표시
print(mask)#대체로 원본데이터가 선택됨
import matplotlib.pyplot as plt
plt.matshow(mask.reshape(1,-1),cmap="gray")#흰색 선택 O, 검은색 선택 x
plt.xlabel("feature num")

#전체와 선택 성능 비교
from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print("whole feature score : {:.3f}".format(lr.score(X_test,y_test)))
lr.fit(X_train_selected,y_train)
print("selected feature score : {:.3f}".format(lr.score(X_test_selected,y_test)))
#특성 선택 경우가 높지만 아닌경우도 있음

##모델기반 선택
#트리모델
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median") #일변량과 비교위해 중간값으로 설정

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape : {}".format(X_train.shape))
print("X_train_l1.shape : {}".format(X_train_l1.shape))

mask = select.get_support()
plt.matshow(mask.reshape(1,-1),cmap="gray")#흰색 선택 O, 검은색 선택 x
plt.xlabel("feature num")

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("test score : {:.3f}".format(score))

##반복적 특성 선택(전진, 후진, 스텝와이즈)
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)

select.fit(X_train, y_train)
mask = select.get_support()
plt.matshow(mask.reshape(1,-1),cmap="gray")#흰색 선택 O, 검은색 선택 x
plt.xlabel("feature num")

X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("test score : {:.3f}".format(score))
print("test score : {:.3f}".format(select.score(X_test,y_test)))#예측

##전문 지식 활용
#자전거 대여소 데이터
import mglearn
import pandas as pd
citibike = mglearn.datasets.load_citibike()
print("data : \n{}".format(citibike.head()))

#대여횟수 스계열 그래프
plt.figure(figsize=(10,3))
xticks = pd.date_range(start=citibike.index.min(),end=citibike.index.max(),freq="D")
week = ["mon","tues","wen","thur","fri","sat","sun"]
xticks_name = [week[int(w)]+d for w, d in zip(xticks.strftime("%w"),
                        xticks.strftime("%m-%d"))]
plt.xticks(xticks, xticks_name, rotation=90, ha="left")
plt.plot(citibike, linewidth=1)
plt.xlabel("date")
plt.ylabel("count")

#posix(기준날(1970.1.1. 00:00)로부터 경과한 초) 시간에 따른 대여횟수
y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1,1)//10**9

#평가+그래프 함수 정의
n_train = 184
def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train,y_train)
    print("test R^2 : {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10,3))
    
    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, label="test")
    plt.plot(range(n_train), y_pred_train, "--", label="train pred")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, "--", label="test pred")
    plt.legend(loc=(1,0))
    plt.xlabel("date")
    plt.ylabel("count")
    
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)
#posix시간으로는 아무것도 되지 않음

#대여시간대만 사용
X_hour = citibike.index.hour.values.reshape(-1,1)
eval_on_features(X_hour, y, regressor)

#요일 추가
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1,1),
                         citibike.index.hour.values.reshape(-1,1)])
eval_on_features(X_hour_week, y, regressor)

#선형회귀로 바꿔보기
from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week, y, LinearRegression())#요일,시간이 정수로 되어 있어 연속형처리

#코딩변경
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
eval_on_features(X_hour_week_onehot, y, LinearRegression())

#상호작용 고려
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, 
                                      include_bias=False, interaction_only=True)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
from sklearn.linear_model import Ridge
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

#특성에 이름 달기
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["mon","tues","wen","thur","fri","sat","sun"]
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_!=0]
coef_nonzero = lr.coef_[lr.coef_!=0]

#학습된 계수
plt.figure(figsize=(15,2))
plt.plot(coef_nonzero, "o")
plt.xticks(np.array(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("feature name")
plt.ylabel("coef size")