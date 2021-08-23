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

# + id="aOBjdDFfWUvS"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline

# + id="gcQ-HVD4WUvT" outputId="8c294a35-4aa3-4ba8-c5f5-c285bc555238"
# cd "/Users/김수빈/Documents/DScover/project_marketing"

# + id="vlLJSQzUWUvU" outputId="796edd81-aab3-4e00-c0e6-ee14632b02ab"
data = pd.read_csv("total.csv", encoding='CP949')
data.head()

# + [markdown] id="hBTM6aU6WUvU"
# Finding optimal k: elbow point

# + id="2BiWMKgOWUvU"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['평균값', 'freq']])
    sse.append(km.inertia_)

# + id="6Pq4aFb3WUvV" outputId="a2ff463b-7710-4053-d0fb-b1cf6ea547d2"
sse

# + id="7Wlk21InWUvV" outputId="523c7c3c-a4f9-40b0-b31a-f3ad7d073bd2"
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="tIKGGjbQWUvV"
# Optimal k: 3

# + id="NyPYZRCdlKMK"


# + [markdown] id="yHB2Bm_YWUvW"
# I. AGE

# + [markdown] id="n6DYaFfgWUvW"
# 1. 10대

# + id="WmjxB1mHWUvW" outputId="2f8496e8-4bab-48e6-e276-c3aee0dfcabd"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['10대', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="vYkr-BfiWUvW"
# Optimal k: 2

# + id="kzHadNLaWUvW" outputId="f4b551e9-cd94-42e8-9db2-a5ca338c708a"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="-PVprQA6WUvX" outputId="10d25f5e-d66a-4c24-e2bf-97b53c838b0a"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['10대'], data1['freq'], color='green')
plt.scatter(data2['10대'], data2['freq'], color='red')

plt.xlabel('10대')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="IP4a0xNaWUvX"
# 2. 20대

# + id="1CkpAtq4WUvX" outputId="ea6c8fc7-0d29-4d7a-b3d2-af599fa7e426"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['20대', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="yNpKjYQOWUvX"
# Optimal k: 2

# + id="-vjfBSFyWUvX" outputId="23e9abc5-6c21-4fa4-ba9e-0e6a77c5bb2f"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="3GCM9lTHWUvY" outputId="769d8884-13da-4a7d-d763-9401f20a780b"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['20대'], data1['freq'], color='green')
plt.scatter(data2['20대'], data2['freq'], color='red')

plt.xlabel('20대')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="_Cn577s8WUvY"
# 3. 30대

# + id="lswhhuYqWUvY" outputId="5e98efd0-6b2b-46cf-bf37-0211cfa2d0ea"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['30대', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="lPv--Du4WUvY"
# Optimal k: 3

# + id="8o9owkSLWUvZ" outputId="92abe103-d70f-4357-c652-43d9eeac1c14"
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="yUu2KqTuWUvZ" outputId="bb99c208-193d-46fb-bbea-d9b474cf4fa5"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]
data3 = data[data['cluster'] == 2]

plt.scatter(data1['30대'], data1['freq'], color='green')
plt.scatter(data2['30대'], data2['freq'], color='red')
plt.scatter(data3['30대'], data3['freq'], color='black')

plt.xlabel('30대')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="cSdKl4jjWUvZ"
# 4. 40대 이상

# + id="u6bMCpzuWUvZ" outputId="9a877005-1229-4a27-d7b2-8ac30c3c3353"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['40대 이상', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="mUmMXR-fWUvZ"
# Optimal k: 5

# + id="0Eq7m9oFWUvZ" outputId="147f9912-eac9-46bc-a69d-fe86fc009cd7"
km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="AU7yCTZNWUva" outputId="6531ad78-c2cf-4bea-8d7c-91ea73c84858"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]
data3 = data[data['cluster'] == 2]
data4 = data[data['cluster'] == 3]
data5 = data[data['cluster'] == 4]

plt.scatter(data1['40대 이상'], data1['freq'], color='green')
plt.scatter(data2['40대 이상'], data2['freq'], color='red')
plt.scatter(data3['40대 이상'], data3['freq'], color='black')
plt.scatter(data4['40대 이상'], data4['freq'], color='blue')
plt.scatter(data5['40대 이상'], data5['freq'], color='yellow')

plt.xlabel('40대 이상')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="RAPHR_uLWUva"
# II. SKINTYPE

# + [markdown] id="8Evuzqi1WUva"
# 1. Oily

# + id="KWEAxg0AWUva" outputId="1efcc52b-728f-48f7-dd7e-f15b6e9a380b"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['Oily', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="ka5Hi3AaWUva"
# Optimal k: 2

# + id="6hstpca1WUva" outputId="f0ad24c9-6111-40d5-e555-ad584155c2bc"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="wZLI7cEqWUvb" outputId="a96d542d-39e4-4a37-deb2-5866cb046607"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['Oily'], data1['freq'], color='green')
plt.scatter(data2['Oily'], data2['freq'], color='red')

plt.xlabel('Oily')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="k9_DKu_TWUvb"
# 2. Dry

# + id="YtkmkCWwWUvb" outputId="ab4a257f-d207-438e-ff86-030a76320d9c"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['Dry', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="0YIi1Uv_WUvb"
# Optimal k: 2

# + id="aea0xsIwWUvb" outputId="312b0653-6dc7-4454-e901-a18c227cdeef"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="RdR5TIldWUvc" outputId="980fe572-1fbe-423a-b934-267491324b58"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['Dry'], data1['freq'], color='green')
plt.scatter(data2['Dry'], data2['freq'], color='red')

plt.xlabel('Dry')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="BqTVvBN6WUvc"
# 3. Sensitive

# + id="oln45RkXWUvc" outputId="22f5cf8d-c411-4034-981d-10cf6d392cff"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['Sensitive', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="Svnerba7WUvc"
# Optimal k: 2

# + id="R8JcwJ4fWUvc" outputId="a759b913-c0e3-4b78-af5d-7da922ad4df3"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="5zu-MdnrWUvc" outputId="216298d2-ee79-47da-d80b-89102ba0dc4f"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['Sensitive'], data1['freq'], color='green')
plt.scatter(data2['Sensitive'], data2['freq'], color='red')

plt.xlabel('Sensitive')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="sU4Uof5hWUvc"
# 4. Normal

# + id="OoUUu4w_WUvd" outputId="a3728d1d-ea43-4359-fb86-41108b52bef7"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['Normal', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="XyfCGKfWWUvd"
# Optimal k: 3

# + id="xo_YwDJeWUvd" outputId="1a9e5e73-737a-472c-a8d7-798de60ec6a3"
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="BesmsxgqWUvd" outputId="68ea52b3-f04e-48c1-9ee3-f22f024f3e35"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]
data3 = data[data['cluster'] == 2]

plt.scatter(data1['Normal'], data1['freq'], color='green')
plt.scatter(data2['Normal'], data2['freq'], color='red')
plt.scatter(data3['Normal'], data3['freq'], color='black')

plt.xlabel('Normal')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="tQg7eMwjWUvd"
# 5. Combination

# + id="eBtdVS5xWUvd" outputId="32477aba-d1c8-4691-c558-ae78a3bf54b9"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['Combination', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="DZJfR7G5WUvd"
# Optimal k: 2

# + id="d0f7ozJSWUve" outputId="6bd676ce-0d5e-4530-a61d-f19182b6b519"
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="TkX7YtWMWUve" outputId="f927da2d-10c4-4a0c-906e-bde04b037f87"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]

plt.scatter(data1['Combination'], data1['freq'], color='green')
plt.scatter(data2['Combination'], data2['freq'], color='red')

plt.xlabel('Combination')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="Vpkttbv0WUve"
# III. RATING

# + id="Mob_s1TuWUve" outputId="4e3a8429-a73a-4012-9b8e-55d467f7edc3"
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k)
    km.fit(data[['평균값', 'freq']])
    sse.append(km.inertia_)
    
    
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

# + [markdown] id="9sEnZ9fXWUve"
# Optimal k: 3

# + id="CzwQkN0rWUve" outputId="42a034c2-4fab-4a8e-b605-0ea34d3fc14d"
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(data[['10대', '20대', '30대', '40대 이상', 'freq', 'Oily', 'Dry', 'Sensitive', 'Normal', 'Combination', '0점', '1점', '2점', '3점', '성분개수', '평균값']])
data['cluster'] = y_predicted
data.head()

# + id="6Ivs0Lc3WUve" outputId="06bdf4a0-e660-4c2f-d470-e6bbae18d2b9"
data1 = data[data['cluster'] == 0]
data2 = data[data['cluster'] == 1]
data3 = data[data['cluster'] == 2]

plt.scatter(data1['평균값'], data1['freq'], color='green')
plt.scatter(data2['평균값'], data2['freq'], color='red')
plt.scatter(data3['평균값'], data3['freq'], color='black')

plt.xlabel('평균값')
plt.ylabel('frequency')
plt.legend()

# + [markdown] id="aoFdRN_RWUve"
# source: https://www.youtube.com/watch?v=EItlUEPCIzM
