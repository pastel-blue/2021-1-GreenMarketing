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

import pandas as pd

from sklearn.linear_model import LogisticRegression

brand = pd.read_csv("C:/Users/geeee/Desktop/data_prefer/innisfree_prefer.csv")

brand

brand['Oily'] = brand['skintype'].apply(lambda x: 1 if x == '지성' else 0)
brand['Dry'] = brand['skintype'].apply(lambda x: 1 if x == '건성' else 0)
brand['Sensitive'] = brand['skintype'].apply(lambda x: 1 if x == '민감성' else 0)
brand['Normal'] = brand['skintype'].apply(lambda x: 1 if x == '중성' else 0)
brand['Combination'] = brand['skintype'].apply(lambda x: 1 if x == '복합성' else 0)

brand['class'] = brand['class'].map({'긍정':1,'부정':0})

preference = brand['class']

features = brand[['age', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination']]

# +
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, preference)

# +
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# +
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_features, train_labels)
# -

print(model.score(train_features, train_labels))
#모델의 정확도

print(model.coef_)
# 변수 중요도(계수)

import numpy as np

# +
#나이별로 피부타입별로 해보기
oily1 = np.array([15.0, 1.0, 0.0, 0.0, 0.0, 0.0])
oily2 = np.array([25.0, 1.0, 0.0, 0.0, 0.0, 0.0])
oily3 = np.array([35.0, 1.0, 0.0, 0.0, 0.0, 0.0])
oily4 = np.array([45.0, 1.0, 0.0, 0.0, 0.0, 0.0])
oily5 = np.array([55.0, 1.0, 0.0, 0.0, 0.0, 0.0])

dry1 = np.array([15.0, 0.0, 1.0, 0.0, 0.0, 0.0])
dry2 = np.array([25.0, 0.0, 1.0, 0.0, 0.0, 0.0])
dry3 = np.array([35.0, 0.0, 1.0, 0.0, 0.0, 0.0])
dry4 = np.array([45.0, 0.0, 1.0, 0.0, 0.0, 0.0])
dry5 = np.array([55.0, 0.0, 1.0, 0.0, 0.0, 0.0])

sensitive1 = np.array([15.0, 0.0, 0.0, 1.0, 0.0, 0.0])
sensitive2 = np.array([25.0, 0.0, 0.0, 1.0, 0.0, 0.0])
sensitive3 = np.array([35.0, 0.0, 0.0, 1.0, 0.0, 0.0])
sensitive4 = np.array([45.0, 0.0, 0.0, 1.0, 0.0, 0.0])
sensitive5 = np.array([55.0, 0.0, 0.0, 1.0, 0.0, 0.0])

normal1 = np.array([15.0, 0.0, 0.0, 0.0, 1.0, 0.0])
normal2 = np.array([25.0, 0.0, 0.0, 0.0, 1.0, 0.0])
normal3 = np.array([35.0, 0.0, 0.0, 0.0, 1.0, 0.0])
normal4 = np.array([45.0, 0.0, 0.0, 0.0, 1.0, 0.0])
normal5 = np.array([55.0, 0.0, 0.0, 0.0, 1.0, 0.0])

combination1 = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 1.0])
combination2 = np.array([25.0, 0.0, 0.0, 0.0, 0.0, 1.0])
combination3 = np.array([35.0, 0.0, 0.0, 0.0, 0.0, 1.0])
combination4 = np.array([45.0, 0.0, 0.0, 0.0, 0.0, 1.0])
combination5 = np.array([55.0, 0.0, 0.0, 0.0, 0.0, 1.0])

sample_users_oily = np.array([oily1, oily2, oily3, oily4, oily5])
sample_users_dry = np.array([dry1, dry2, dry3, dry4, dry5])
sample_users_sensitive = np.array([sensitive1, sensitive2, sensitive3, sensitive4, sensitive5])
sample_users_normal = np.array([normal1, normal2, normal3, normal4, normal5])
sample_users_combination = np.array([combination1, combination2, combination3, combination4, combination5])
# -

sample_users_oily = scaler.transform(sample_users_oily)
sample_users_dry = scaler.transform(sample_users_dry)
sample_users_sensitive = scaler.transform(sample_users_sensitive)
sample_users_normal = scaler.transform(sample_users_normal)
sample_users_combination = scaler.transform(sample_users_combination)

print(model.predict(sample_users_oily))
print(model.predict(sample_users_dry))
print(model.predict(sample_users_sensitive))
print(model.predict(sample_users_normal))
print(model.predict(sample_users_combination))

print(model.predict_proba(sample_users_oily))
print(model.predict_proba(sample_users_dry))
print(model.predict_proba(sample_users_sensitive))
print(model.predict_proba(sample_users_normal))
print(model.predict_proba(sample_users_combination))

oily_result = print(model.predict_proba(sample_users_oily))

sns.scatterplot(x="age", y="oily_result", data=brand)
plt.grid(True)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(brand,hue="Oily",vars=['age','rating_new'])

sns.pairplot(brand,hue="Oily",vars=['age','rating_new'], kind="reg", size=3)
plt.show

sns.scatterplot(x="age", y="class_new", data=brand)
plt.grid(True)
plt.show()

sns.scatterplot(x="age", y="rating_new", data=brand)
plt.grid(True)
plt.show()

plt.plot(brand['age'],brand['rating_new'],'x')

m,b = np.polyfit(brand['age'], brand['rating_new'],1)

plt.plot(brand['age'],m*brand['age']+b)
plt.show()

sns.scatterplot(x="Oily", y="rating_new", data=brand)
plt.grid(True)
plt.show()


