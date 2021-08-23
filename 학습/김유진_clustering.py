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
import numpy as np 
import re
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df = pd.read_csv("/Users/yujinkim/Desktop/DScover/마케팅팀_프로젝트/total.csv",encoding = 'cp949')
df 

new_df = df.iloc[:, 3:]

new_df

age_df = new_df.iloc[:, :5]
age_df

skintype_df = new_df.iloc[:, 4:10]
skintype_df

from matplotlib import font_manager, rc
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

# +
from pandas.plotting import scatter_matrix

sns.pairplot(data = skintype_df)
# -

sns.pairplot(data = skintype_df)

# +
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
# -

#스케일링X, 클러스터링 시도
X = skintype_df.drop(['freq'], axis =1)
Y = skintype_df['freq']

k_range = range(1,11)

k_means_models = [KMeans(n_clusters=k, random_state=1).fit(X) for k in k_range]

centroids = [one_model.cluster_centers_ for one_model in k_means_models]
centroids

#유클리드 거리 계산
k_euclid =[cdist(X, cent, 'euclidean') for cent in centroids]

dist = [np.min(ke, axis=1) for ke in k_euclid]

len(dist)

len(dist[1])

k_means_2 = KMeans(n_clusters=2, random_state=1).fit(X)
k_means_3 = KMeans(n_clusters=3, random_state=1).fit(X)
k_means_4 = KMeans(n_clusters=4, random_state=1).fit(X)

metrics.silhouette_score(X, k_means_2.labels_, metric='euclidean')

metrics.silhouette_score(X, k_means_3.labels_, metric='euclidean')

metrics.silhouette_score(X, k_means_4.labels_, metric='euclidean')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# k에 따라 inertia_(군집 내 거리제곱합의 합)이 어떻게 변하는 지 시각화
def change_n_clusters(n_clusters, data):
    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
        
    plt.figure(1 , figsize = (12, 6))
    plt.plot(n_clusters , sum_of_squared_distance , 'o')
    plt.plot(n_clusters , sum_of_squared_distance , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')



change_n_clusters(k_range, age_df)

# +
#스케일링 하지 않은 데이터
kmeans = KMeans(n_clusters=2)
kmeans.fit(age_df)

plt.figure(figsize=(20, 20))
plt.subplot(2,2,1)
sns.scatterplot(x='10대', y='freq', data=age_df, hue=kmeans.labels_,palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)

plt.subplot(2, 2, 2)
sns.scatterplot(x='20대', y='freq', data=age_df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)

plt.subplot(2, 2, 3)
sns.scatterplot(x='30대', y='freq', data=age_df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', alpha=0.5, s=150)

plt.subplot(2, 2, 4)
sns.scatterplot(x='40대 이상', y='freq', data=age_df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', alpha=0.5, s=150)

# -

#outlier 제거하기
def outlier_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3-q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return np.where((data>upper_bound)|(data<lower_bound))


freq_outlier_index = outlier_iqr(age_df)

nooutlier_age_df = age_df.loc[freq_outlier_index, '']
