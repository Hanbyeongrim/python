# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:40:02 2018

@author: A
"""

from sklearn.datasets import load_files
reviews_train = load_files("C:/Users/A/.spyder-py3/aclImdb/train/")
text_train, y_train = reviews_train.data, reviews_train.target
print("text_train type : {}".format(type(text_train)))
print("text_train length : {}".format(len(text_train)))
print("text_train[6] : \n{}".format(text_train[6]))
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
import numpy as np
print("class sample : {}".format(np.bincount(y_train)))

reviews_test = load_files("C:/Users/A/.spyder-py3/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("text_test length : {}".format(len(text_test)))
print("class sample : {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

##BOW
bards_words = ["The fool doth think he is wise,",
               "but the wise man knows himself to be a fool"]
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)
print("word dic size : {}".format(len(vect.vocabulary_)))
print("word dic text : {}".format(vect.vocabulary_))

bag_of_words = vect.transform(bards_words)
print("bow : {}".format(repr(bag_of_words)))
print("bow의 밀집 표현 : \n{}".format(bag_of_words.toarray()))

#리뷰데이터에 적용
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X_train : \n{}".format(repr(X_train)))

feature_names = vect.get_feature_names()
print("feature count : {}".format(len(feature_names)))
print("first~20 : \n{}".format(feature_names[:20]))#의미없는것들도 포함
print("20010~20030 : \n{}".format(feature_names[20010:20030]))#단복수가 같이 들어감
print("2000n : \n{}".format(feature_names[::2000]))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("cross val score : {:.2f}".format(np.mean(scores)))

from sklearn.model_selection import GridSearchCV
param_grid = {"C":[0.001, 0.01, 0.1, 1 ,10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("best cross val score : {:.2f}".format(grid.best_score_))
print("best param : {}".format(grid.best_params_))
X_test = vect.transform(text_test)
print("test score : {:.2f}".format(grid.score(X_test, y_test)))

vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("min_df X_train : {}".format(repr(X_train)))
feature_names = vect.get_feature_names()
print("first~50 : \n{}".format(feature_names[:50]))
print("20010~20030 : \n{}".format(feature_names[20010:20030]))
print("700n : \n{}".format(feature_names[::700]))
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("best cross val score : {:.2f}".format(grid.best_score_))

##불용어:의미없거나 너무 빈번해 유용하지 않은 단어
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("stopword count : {}".format(len(ENGLISH_STOP_WORDS)))#318개
print("10n stopword : \n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)#내장된 영어 불용어
X_train = vect.transform(text_train)
print("stopword without X_train : \n{}".format(repr(X_train)))#27271->26966 305개 감소

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {"C":[0.001, 0.01, 0.1, 1 ,10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("best cross val score : {:.2f}".format(grid.best_score_))#변화작지만 감소.
#고정불용어는 작은데이터에 적합

##tf-idf : 특정문서 자주나타나는 단어에 가중치
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {"logisticregression__C" : [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("best cross val score : {:.2f}".format(grid.best_score_))

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
X_train = vectorizer.transform(text_train)
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
import numpy as np
feature_names = np.array(vectorizer.get_feature_names())
print("low tfidf feature : \n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("hihg tfidf feature : \n{}".format(feature_names[sorted_by_tfidf[-20:]]))

sorted_by_idf = np.argsort(vectorizer.idf_)
print("lowest idf feature : \n{}".format(feature_names[sorted_by_idf[:100]]))
#the, and, of 처럼 자주 쓰는 단어들 등장
#good, bad : 감정분석에서는 중요하겠지만 tfidf에서는 안 중요

##모델 계수
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps
                                     ["logisticregression"].coef_[0], feature_names, 
                                     n_top_features=40)
#음수계수는 부정적 리뷰, 양수계수는 긍정적 리뷰

##여러토큰 bow : 맥락 고려 위함
print("bards_words : \n{}".format(bards_words))
cv = CountVectorizer(ngram_range=(1,1)).fit(bards_words)#토큰 1개짜리만
print("word dic size : {}".format(len(cv.vocabulary_)))
print("word dic : {}".format(cv.get_feature_names()))

cv = CountVectorizer(ngram_range=(2,2)).fit(bards_words)#토큰 2개짜리만
print("word dic size : {}".format(len(cv.vocabulary_)))
print("word dic : {}".format(cv.get_feature_names()))
print("transform data : \n{}".format(cv.transform(bards_words).toarray()))

cv = CountVectorizer(ngram_range=(1,3)).fit(bards_words)#토큰 1~3개
print("word dic size : {}".format(len(cv.vocabulary_)))
print("word dic : {}".format(cv.get_feature_names()))

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {"logisticregression__C" : [0.001, 0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range" : [(1,1),(1,2),(1,3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("best cross val score : {:.2f}".format(grid.best_score_))
print("best param : {}".format(grid.best_params_))

#ngram 정확도 히트맵
import matplotlib.pyplot as plt
scores = grid.cv_results_["mean_test,score"].reshape(-1,3).T
heatmap = mglearn.tools.heatmap(scores, xlabel="C", ylabel="ngram_range",cmap="viridis",
                                fmt="%.3f", 
                                xticklabels=param_grid["logisticregression__C"],
                                yticklabels=param_grid["tfidfvectorizer__ngram_range"])
plt.colorbar(heatmap)

#1~3그램 모델에서 가장 중요한 특성
vect = grid.best_estimator_.named_steps["tfidfvectorizer"]
featrue_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps["logisticregression"].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
#3그램
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
mglearn.tools.visualize_coefficients(coef.rabel()[mask],feature_names[mask],
                                     n_top_features=40)

##어간추출, 표제어 추출. ##버전문제로 spacy, konlpy 설치가 안됨
import spacy #어간
import nltk #표제어

en_nlp = spacy.load("en")#영어모델
stemmer = nltk.stem.PorterStemmer()

def compare_normalization(doc):
    doc_spacy = en_nlp(doc)#spacy로 토큰화
    print("표제어 : ")
    print([token.lemma_ for token in doc_spacy])
    print("어간 : ")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

compare_normalization(u"Our meeting today was worse than yerterday, "
                      "I'm scared of metting the clients tomorrow.")

import re
regexp = re.compile('(?u)\\b\\w\\w+\\b')
en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
        regexp.findall(string))

def custom_tokenizer(document): #커스텀 토큰 분할기
    doc_spacy = en_nlp(document, entity=False, parse=False)
    return [token.lemma_ for token in doc_spacy]
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

X_train_lemma = lemma_vect.fit_trainsform(text_train)
print("X_train_lemma.shape : {}".format(X_train_lemma.shape))

vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train.shape : {}".format(X_train.shape))

#그리드서치
from sklearn.model_selection import StratifiedShuffleSplit
param_grid = {"C":[0.001, 0.01, 0.1, 1, 10]}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99, train_size=0.01, random_state=0)
grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
grid.fit(X_train, y_train)
print("best cross val score (기본CV) : {:.2f}".format(grid.best_score_))
grid.fit(X_train_lemma, y_train)
print("best corss val score (표제어) : {:.2f}".format(grid.best_score_))

###########konlpy######

##LDA 잠재 디리클레 할당 : 비지도학습으로 문서를 토픽으로 할당. 군집화의 일종
#자주나오는 단어 제거하는게 좋음 적어도 15%
vect = CountVectorizer(max_features=10000, max_df=0.15)
X = vect.fit_transform(text_train)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10, learning_method="batch", 
                                max_iter=25, random_state=0)
document_topics = lda.fit_transform(X) #시간 오래 걸림
print("lda.components_.shape : {}".format(lda.components_.shape))

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=10)
#할당된 문서의 높은순위 단어 확인

lda100 = LatentDirichletAllocation(n_topics=100, learning_method="batch", 
                                max_iter=25, random_state=0)#토픽 많을수록 복잡하지만 유의함
document_topics100 = lda100.fit_transform(X)

topics = np.array([7,16,24,25,28,36,37,45,51,53,54,63,89,97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=7, n_words=20)

#45번이 음악에 관한것인듯
music = np.argsort(document_topics100[:,45])[::-1]
for i in music[:10]:
    print(b".".join(text_train[i].split(b".")[:2])+b".\n")
    
#토픽 가중치
fig, ax = plt.subplits(1, 2, figsize=(10,10))
topic_names = ["{:2} ".format(i) + " ".join(words) #각 토픽을 대표하는 두단어 합치기
                for i, words in enumerate(feature_names[sorting[:, :2]])]
for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabel(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()