# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:03:40 2018

@author: A
"""

##정확도
#불균형 데이터
from sklearn.datasets import load_digits

digits = load_digits()
y = digits.target == 9

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        digits.data, y, random_state=0)

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

import numpy as np
print("predict unique label : {}".format(np.unique(pred_most_frequent)))
print("test score : {:.2f}".format(dummy_majority.score(X_test, y_test)))
#특별한 학습을 하지 않았지만 정확도가 0.9가 나옴. > 문제 있음

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("test score : {:.2f}".format(tree.score(X_test, y_test)))#더미모델과 큰 차이 없음

from sklearn.linear_model import LogisticRegression
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score : {:.2f}".format(dummy.score(X_test, y_test)))
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score : {:.2f}".format(logreg.score(X_test, y_test)))
#더미모델의 성능도 괜찮아 유용성 판단 어려움

#오차행렬
import mglearn
mglearn.plots.plot_confusion_matrix_illustration()
mglearn.plots.plot_binary_confusion_matrix()

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_logreg)
print("오차행렬 : \n{}".format(confusion))

print("빈도기반 더미모델")
print(confusion_matrix(y_test, pred_most_frequent))#하나로만 예측함
print("\n무작위 더미모델")
print(confusion_matrix(y_test, pred_dummy))
print("\n결정나무")
print(confusion_matrix(y_test, pred_tree))#무작위더미보다 나음
print("\n로지스틱 회귀")
print(confusion_matrix(y_test, pred_logreg))#트리보다 나음

#정확도 지표
#precision 정밀도, recall 재현율
from sklearn.metrics import f1_score
print("빈도 기반 더미 모델의 f1 : {:.2f}".format(f1_score(y_test, pred_most_frequent)))
print("무작위 더미 모델의 f1 : {:.2f}".format(f1_score(y_test, pred_dummy)))
print("트리 모델의 f1 : {:.2f}".format(f1_score(y_test, pred_tree)))
print("로지스틱 회귀 모델의 f1 : {:.2f}".format(f1_score(y_test, pred_logreg)))

from sklearn.metrics import classification_report #결과의 마지막 줄은 가중 평균
print(classification_report(y_test, pred_most_frequent, target_names=["not 9","9"]))
print(classification_report(y_test, pred_dummy, target_names=["not 9","9"]))
print(classification_report(y_test, pred_logreg, target_names=["not 9","9"]))

#불확실성 고려 : 임계값 조정
import mglearn
from mglearn.datasets import make_blobs
X, y = make_blobs(n_samples=(400,50), centers=2, cluster_std=[7.0, 2],
                  random_state=22)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.svm import SVC
svc = SVC(gamma=0.05).fit(X_train, y_train)

mglearn.plots.plot_decision_threshold()
from sklearn.metrics import classification_report
print(classification_report(y_test, svc.predict(X_test)))

y_pred_lower_threshold = svc.decision_function(X_test) > -0.8
print(classification_report(y_test, y_pred_lower_threshold))

##정밀도 재현율 곡선
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))#가능한 모든 임계값에서 정밀도, 재현율 반환

X, y = make_blobs(n_samples=(4000,500), centers=2, cluster_std=[7, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=0.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))

import numpy as np
close_zero = np.argmin(np.abs(thresholds))

import matplotlib.pyplot as plt
plt.plot(precision[close_zero],recall[close_zero], 'o', markersize=10,
         label="hold 0", fillstyle="none", c='k', mew=2)#곡선이 오른쪽 위로 치우친게 좋은것
plt.plot(precision, recall, label="roc")
plt.xlabel("pre")
plt.ylabel("recall")
plt.legend(loc="best")

##svc, rf 비교 : 임계값에 따라 좋은 모델이 다름, f1값만으로는 알 수 없는 부분.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
        y_test, rf.predict_proba(X_test)[:,1])
plt.plot(precision, recall, label="svc")

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="svc: hold 0", fillstyle="none", c='k', mew=2)
plt.plot(precision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', markersize=10,
         label="rf: hold 0.5", fillstyle="none", c='k', mew=2)
plt.plot(precision_rf, recall_rf, label="rf")
plt.xlabel("pre")
plt.ylabel("recall")
plt.legend(loc="best")

from sklearn.metrics import f1_score
print("랜덤 포레스트 f1 : {:.3f}".format(f1_score(y_test, rf.predict(X_test))))
print("svc f1 : {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

#평균 정밀도 : 정밀도 누적값을 양성 샘플 수로 나눔. roc의 아랫부분 면적과 유사
from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:,1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("rf avg pre : {:.3f}".format(ap_rf))
print("svc avg pre : {:.3f}".format(ap_svc))#둘이 비슷

##roc, auroc
#tpr : 재현율, fpr : 특이도(거짓 분류 중에 진짜 거짓)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="roc curve")
plt.xlabel("fpr")
plt.ylabel("tpr(recall)")

close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', 
         markersize=10, label="hold 0", fillstyle="none", c='k', mew=2)#svc roc
plt.legend(loc="best")

#svc, rf roc 비교
from sklearn.metrics import roc_curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label="svc roc")
plt.plot(fpr_rf, tpr_rf, label="rf roc")
plt.xlabel("fpr")
plt.ylabel("tpr(recall)")

plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="svc: hold 0", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr_rf[close_default_rf], '^', markersize=10,
         label="rf: hold 0.5", fillstyle="none", c='k', mew=2)
plt.legend(loc="best")
#auroc : 불균형 데이터셋에서는 정확도보다 좋은 지표. 그러나 임계값 조정이 필요
from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("rf auc : {:.3f}".format(rf_auc))
print("svc auc : {:.3f}".format(svc_auc))

#digit data
from sklearn.datasets import load_digits
digits = load_digits()

y = digits.target == 9
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        digits.data, y, random_state=0)

plt.figure()#정확도는 같으나 auc값이 다름.
for gamma in [1, 0.1, 0.01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
    print("gamma = {:.2f}, accuracy = {:.2f}, auc = {:.2f}".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.01)
plt.legend(loc="best")

##다중 분류 지표
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("accuracy : {:.3f}".format(accuracy_score(y_test, pred)))
print("confusion matrix : \n{}".format(confusion_matrix(y_test, pred)))

#오차행렬 시각화
import matplotlib.pyplot as plt
import mglearn
scores_image = mglearn.tools.heatmap(
        confusion_matrix(y_test, pred), xlabel="pred", ylabel="real", 
        xticklabels=digits.target_names, yticklabels=digits.target_names, 
        cmap=plt.cm.gray_r, fmt="%d")
plt.title("confusion matrix")
plt.gca().invert_yaxis()

#정밀도 재현율 f1
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#다중분류의 f1 : 클래스별 f1값 평균을 3가지 방법으로 구함(macro, weighted, micro)
#샘플이 같다면 micro, 클래스 수 같다면 macro
from sklearn.metrics import f1_score
print("micro mean f1 : {:.3f}".format(f1_score(y_test, pred, average="micro")))
print("macro mean f1 : {:.3f}".format(f1_score(y_test, pred, average="macro")))

##모델 선택에 지표 사용하기
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
print("basic index : {}".format(cross_val_score(SVC(),digits.data, digits.target == 9)))
explicit_accuracy = cross_val_score(SVC(), digits.data, 
                                    digits.target == 9, scoring="accuracy")#정확도로 선택
print("accuracy index : {}".format(explicit_accuracy))
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc")
print("auc index : {}".format(roc_auc))#기본값이 accuracy, 문자열로 roc_auc 지정

#그리드 서치
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=0)
param_grid = {"gamma" : [0.0001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("grid search accuracy index")
print("best param : ", grid.best_params_)
print("best cross_val score : {:.3f}".format(grid.best_score_))
print("test set auc : {:.3f}".format(roc_auc_score(
        y_test, grid.decision_function(X_test))))
print("test set score : {:.3f}".format(grid.score(X_test, y_test)))

grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)#오류 발생. 검색해보니 y에 인코딩이 필요한듯.
print("grid search roc_auc index")
print("best param : ", grid.best_params_)
print("best cross_val score : {:.3f}".format(grid.best_score_))
print("test set auc : {:.3f}".format(roc_auc_score(
        y_test, grid.decision_function(X_test))))
print("test set score : {:.3f}".format(grid.score(X_test, y_test)))

from sklearn.metrics.scorer import SCORERS
print("가능한 평가 방식 : \n{}".format(sorted(SCORERS.keys())))