# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import LinearSVC

os.chdir('\\Users\\82105\\Desktop\\SKKU\\prefer\\data_prefer')
os.getcwd()

# # 하나의 브랜드 예시: Lush

lush = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/lush_prefer.csv", engine='python')

lush

# +
# "특정 고객이 해당 브랜드 제품에 긍정/부정적 의견을 보일 것이다", 분류 용이하게 하기 위해 class열을 추가해 긍정을 1, 부정은 0 값 부여

lush['class'] = lush['class'].map({'긍정':1,'부정':0})

# +
# 5가지의 피부타입(건성, 지성...)을 0 ~ 4 사이의 정수로 라벨링
# 0: 건성, 1: 민감성, 2: 복합성, 3: 중성, 4: 지성

le = preprocessing.LabelEncoder()
skintype_encoded = le.fit_transform(lush["skintype"])
print(skintype_encoded)

# +
# feature를 특정 고객의 피부타입과 연령 조합으로 설정하고 그 고객이 해당 브랜드에 갖는 긍정 or 부정 의견을 label로 설정

features = list(zip(skintype_encoded, lush["age"]))
label = lush["class"]
print(features)

# +
# 사용예시) 피부타입이 0(건성)인 25세 고객이 lush 브랜드 제품에 부정(0) 의견을 보일 것이다

model = KNeighborsClassifier(n_neighbors=3)
model.fit(features, label)

predicted = model.predict([[0,25]])
print(predicted)
# -

# ### neighbor가 몇일 때 가장 정확도가 높은지 plot으로 확인하기
# 참고 https://www.datacamp.com/community/tutorials/introduction-machine-learning-python

# +
# 데이터셋을 테스트셋과 학습셋으로 나눔

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

# +
# neighbor의 크기가 1부터 8 사이 정수일 때 갖는 정확도를 변수에 저장

neighbors = np.arange(1,9)
train_accuracy =np.zeros(len(neighbors))
test_accuracy = np.zeros(len(neighbors))


for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

# +
# 각 정수일 때 갖는 정확도의 크기를 plot으로 나타냄

plt.figure(figsize=(10,6))
plt.title('KNN accuracy with varying number of neighbors',fontsize=20)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend(prop={'size': 20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# -

# ### -> neighbors = 3 또는 5, 7 일 때 정확도가 가장 높음 
# 매번 정확한 정확도는 바뀌는 듯 함

# +
# neighbor의 크기가 7일 때의 정확도

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# -

# #### 아래는 classification report (해석은 잘 못하겠으나 그리 좋지 않은 결과인 건 알겠음..)

print(classification_report(y_test, y_pred))

# ### age를 연령대로 묶은 상태에서 정확도 계산
# 10대, 20대, 30대, 40대 이상을 각각 0, 1, 2, 3으로

# +
# 연령대로 묶는 작업

lush_demo = lush

for i in range(5251):
    if lush_demo["age"][i] >= 19:
        lush_demo["age"][i] = int(0)
    elif lush_demo["age"][i] >= 20 and lush_demo["age"][i] <= 29:
        lush_demo["age"][i] = int(1)
    elif lush_demo["age"][i] >= 30 and lush_demo["age"][i] <= 39:
        lush_demo["age"][i] = int(2)
    else:
        lush_demo["age"][i] = int(3)

# +
skintype_encoded_ = le.fit_transform(lush_demo["skintype"])
print(skintype_encoded_)

features_ = list(zip(skintype_encoded_, lush_demo["age"]))
label_ = lush_demo["class"]

X_train_, X_test_, y_train_, y_test_ = train_test_split(features_, label_, test_size=0.3)

# +
neighbors_ = np.arange(1,9)
train_accuracy_ =np.zeros(len(neighbors))
test_accuracy_ = np.zeros(len(neighbors))

for i,k in enumerate(neighbors_):
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(X_train_, y_train_)

    #Compute accuracy on the training set
    train_accuracy_[i] = knn.score(X_train_, y_train_)

    #Compute accuracy on the test set
    test_accuracy_[i] = knn.score(X_test_, y_test_)
# -

plt.figure(figsize=(10,6))
plt.title('KNN accuracy with varying number of neighbors (Age group)',fontsize=20)
plt.plot(neighbors_, test_accuracy_, label='Testing Accuracy')
plt.plot(neighbors_, train_accuracy_, label='Training accuracy')
plt.legend(prop={'size': 20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# ### -> neighbors =  7일 때 정확도 높음 (그때그때 달라질 수 있음)

# +
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train_, y_train_)
y_pred_ = knn.predict(X_test_)

print("Accuracy:",metrics.accuracy_score(y_test_, y_pred_))
# -

# ###### -> 연령을 군집화했을 때 (10대, 20대 등으로 묶어 0 ~ 3 숫자 부여)가  그렇지 않을 때 (27세, 18세 등)보다 정확도가 좀 더 높음! (자세히는 모르지만 숫자를 정규화 했을 때 정확도가 더 높아진다고 들었음. 정규화의 원래 방향은 0과 1 사이의 숫자로 나타내는 것이지만 그냥 0~3으로 설정함)

print(classification_report(y_test_, y_pred_))

# # all brands

total = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/total_prefer.csv", engine='python')

total

# +
# "특정 고객이 해당 브랜드 제품에 긍정/부정적 의견을 보일 것이다", 분류 용이하게 하기 위해 class열을 추가해 긍정을 1, 부정은 0 값 부여

total['class'] = total['class'].map({'긍정':1,'부정':0})

# +
# 5가지의 피부타입(건성, 지성...)을 0 ~ 4 사이의 정수로 라벨링
# 0: 건성, 1: 민감성, 2: 복합성, 3: 중성, 4: 지성

le = preprocessing.LabelEncoder()
skintype_encoded = le.fit_transform(total["skintype"])
print(skintype_encoded)

# +
# 9개의 브랜드를 0 ~ 8 사이의 정수로 라벨링
# 0: abib, 1 : aveda, 2: innisfree, 3: kiehl, 4: lush, 5: melixir, 6: origins, 7: roundlab, 8: thebodyshop

brand_encoded = le.fit_transform(total["brand"])
print(brand_encoded)

# +
# feature를 (브랜드와 특정 고객의 피부타입, 연령) 조합으로 설정하고 그 고객이 해당 브랜드에 갖는 긍정 or 부정 의견을 label로 설정

features = list(zip(brand_encoded, skintype_encoded, total["age"]))
label = total["class"]

# +
# 사용예시) 피부타입이 1(민감성)이고 나이가 29세인 고객은 이니스프리 브랜드 제품에 부정(2) 의견을 보일 것이다

model = KNeighborsClassifier(n_neighbors=7)
model.fit(features, label)

predicted = model.predict([[2,1,29]])
print(predicted)

# +
# 데이터셋을 테스트셋과 학습셋으로 나눔

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

# +
# 정확도가 가장 높은 neighbor 크기 찾기
# neighbor의 크기가 1부터 8 사이 정수일 때 갖는 정확도를 변수에 저장

neighbors = np.arange(1,9)
train_accuracy =np.zeros(len(neighbors))
test_accuracy = np.zeros(len(neighbors))


for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)
# -

plt.figure(figsize=(10,6))
plt.title('KNN accuracy with varying number of neighbors',fontsize=20)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend(prop={'size': 20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# ### -> neighbors = 7일 때 정확도 높음 (그때그때 달라질 수 있음)

# +
# neighbor의 크기가 7일 때 정확도 

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# -

print(classification_report(y_test, y_pred))

# ### age를 연령대로 묶은 상태에서 정확도 계산
# 10대, 20대, 30대, 40대 이상을 각각 0, 1, 2, 3으로

# +
total_demo = total

for i in range(5251):
    if total_demo["age"][i] >= 19:
        total_demo["age"][i] = int(0)
    elif total_demo["age"][i] >= 20 and total_demo["age"][i] <= 29:
        total_demo["age"][i] = int(1)
    elif total_demo["age"][i] >= 30 and total_demo["age"][i] <= 39:
        total_demo["age"][i] = int(2)
    else:
        total_demo["age"][i] = int(3)

# +
skintype_encoded_ = le.fit_transform(total_demo["skintype"])
print(skintype_encoded_)

brand_encoded_ = le.fit_transform(total_demo["brand"])
print(brand_encoded_)

features_ = list(zip(brand_encoded_, skintype_encoded_, total_demo["age"]))
label_ = total_demo["class"]
# -

X_train_, X_test_, y_train_, y_test_ = train_test_split(features_, label_, test_size=0.3)

# +
neighbors_ = np.arange(1,9)
train_accuracy_ =np.zeros(len(neighbors))
test_accuracy_ = np.zeros(len(neighbors))

for i,k in enumerate(neighbors_):
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(X_train_, y_train_)

    #Compute accuracy on the training set
    train_accuracy_[i] = knn.score(X_train_, y_train_)

    #Compute accuracy on the test set
    test_accuracy_[i] = knn.score(X_test_, y_test_)
# -

plt.figure(figsize=(10,6))
plt.title('KNN accuracy with varying number of neighbors (Age group)',fontsize=20)
plt.plot(neighbors_, test_accuracy_, label='Testing Accuracy')
plt.plot(neighbors_, train_accuracy_, label='Training accuracy')
plt.legend(prop={'size': 20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# +
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train_, y_train_)
y_pred_ = knn.predict(X_test_)

print("Accuracy:",metrics.accuracy_score(y_test_, y_pred_))
