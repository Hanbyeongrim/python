# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:50:23 2018

@author: A
"""

##var transform
##문자데이터 가변수화
#adult data
import os
import pandas as pd
import mglearn
from IPython.display import display

data = pd.read_csv(
        os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),header=None, index_col=False,
        names=["age","workclass","fnlwgt","education","deucation-num","marital-statu",
               "occupation","relationship","race","gender","capital-gain","capital-loss",
               "hour-per-week","native-country","incom"])
data = data[["age","workclass","education","gender","hour-per-week","occupation","incom"]]
display(data.head())#head로 0~4행(5개 행)만 보여줌

print(data.gender.value_counts())

print("original character :\n",list(data.columns),"\n")
data_dummies = pd.get_dummies(data)#get_dummies : 더미화 시키는 함수
print("after get_dummies character : \n",list(data_dummies.columns))#연속형은 그대로.

data_dummies.head()

#values로 numpy 배열화(학습을 위한 데이터프레임으로 바꾸기)
features = data_dummies.loc[:,"age":"occupation_ Transport-moving"]
X = features.values
y = data_dummies["incom_ >50K"].values
print("X.shape : {} / y.shape : {}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("test score : {:.2f}".format(logreg.score(X_test,y_test)))

#숫자데이터 가변수화
#random data
demo_df = pd.DataFrame({"number feature":[0,1,2,1],
                        "range feature":["cat","book","cat","dog"]})
display(demo_df)

display(pd.get_dummies(demo_df))

demo_df["number feature"] = demo_df["number feature"].astype(str) #데이터 형 변환
display(pd.get_dummies(demo_df, columns=["number feature", "range feature"]))
#columns옵션으로 변환할 변수 지정(문자로 형변환 했다면 지정 안해도 됨)


##구간 분할
#wave data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line),label="decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line),"--", label="linear regression")

plt.plot(X[:,0],y,"o",c="k")
plt.ylabel("regressor output")
plt.xlabel("input")
plt.legend(loc="best")

#구간 나누기
bins = np.linspace(-3,3,11)#구간 10개
print("section : {}".format(bins))#구간 나누는 포인트

which_bin = np.digitize(X, bins=bins)#각 데이터 포인트가 어느 구간에 속하는지 기록
print("point :\n",X[:5])
print("section of point : \n",which_bin[:5])

#코딩 변환 onehotencoder는 숫자로된 범주형에만 적용
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)

X_binned = encoder.transform(which_bin)
print(X_binned[:5])
print("X_binned shape : {}".format(X_binned.shape))

#변환한 데이터를 모델에 적용
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned),label="linear regression")

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned),"--", label="decision tree")

plt.plot(X[:,0],y,"o",c="k")
plt.vlines(bins,-3,3,linewidth=1,alpha=0.2)
plt.ylabel("regressor output")
plt.xlabel("input")
plt.legend(loc="best")

##상호작용, 다항식 추가
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label="bined + origin reg")

for bin in bins:
    plt.plot([bin, bin],[-3, 3],":", c="k", linewidth=1)
plt.legend(loc="best")
plt.ylabel("reg print")
plt.xlabel("input")
plt.plot(X[:,0],y,"o",c="k")

#상호작용 추가
X_product = np.hstack([X_binned, X*X_binned])
print(X_product.shape)

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label="bined * origin reg")

for bin in bins:
    plt.plot([bin, bin],[-3, 3],":", c="k", linewidth=1)
plt.legend(loc="best")
plt.ylabel("reg print")
plt.xlabel("input")
plt.plot(X[:,0],y,"o",c="k")

#다항식 추가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10, include_bias=False)#degree=n : X^n차항까지 추가
poly.fit(X)
X_poly = poly.transform(X)
print("X_poly.shape : {}".format(X_poly.shape))

print("X element :\n{}".format(X[:5]))
print("X_poly element : \n{}".format(X_poly[:5]))
print("section name :{}".format(poly.get_feature_names()))

#다항식 추가한 회귀(단 : 데이터 없는 곳에선 과적합)
reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label="multinomial reg")
plt.legend(loc="best")
plt.ylabel("reg print")
plt.xlabel("input")
plt.plot(X[:,0],y,"o",c="k")
#원본 데이터svm과 비교
from sklearn.svm import SVR
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label="SVR gamma={}".format(gamma))
plt.legend(loc="best")
plt.ylabel("svr print")
plt.xlabel("input")
plt.plot(X[:,0],y,"o",c="k")

##보스턴 집값 데이터(표준화 + 다항식 추가)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#poly에 interaction_only=True하면 제곱항 제외됨
poly = PolynomialFeatures(degree=2, include_bias=False).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape : {}".format(X_train.shape))
print("X_train_poly.shape : {}".format(X_train_poly.shape))
print("section name :{}".format(poly.get_feature_names()))

#ridge
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)
print("interaction X score : {:.3f}".format(ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly, y_train)
print("interaction O score : {:.3f}".format(ridge.score(X_test_poly, y_test)))
#상호작용 있을때가 높음

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
print("interaction X score : {:.3f}".format(rf.score(X_test_scaled, y_test)))
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_poly, y_train)
print("interaction O score : {:.3f}".format(ridge.score(X_test_poly, y_test)))
#상호작용 없을때가 높음. 그러나 상호작용 있는 것은 릿지와 값이 거의 비슷

##일변량 변환(log, exp)
#카운트 데이터
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000,3))#1000행 3열 노멀 데이터
w = rnd.normal(size=3)

X = rnd.poisson(10*np.exp(X_org))
y = np.dot(X_org, w)
print(X[:10,0])

print("count : {}".format(np.bincount(X[:,0])))

bins = np.bincount(X[:,0]) #X[:,1] X[:,2] 도 분포가 비슷
plt.bar(range(len(bins)),bins, color="grey")
plt.ylabel("count")
plt.xlabel("num")

#바로 릿지 적용
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
rg_score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("test score : {:.3f}".format(rg_score))#R^2값이 낮음

#log적용하기(데이터에 0이 있어 X+1에 적용)
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.hist(X_train_log[:,0], bins=25, color="gray")
plt.ylabel("count")
plt.xlabel("num")

rg_score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("test score : {:.3f}".format(rg_score))#R^2값이 높아졌음