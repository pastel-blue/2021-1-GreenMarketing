#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[9]:


cd /Users/stella/Documents/DScover/


# In[4]:


data = pd.read_csv('total.csv', encoding='cp949')


# In[6]:


data.head()


# In[6]:


data


# In[5]:


data.drop('Column2', axis=1, inplace=True)


# In[8]:


data.info()


# In[9]:


data['brand'].value_counts()


# In[10]:


data['제품'].value_counts()


# In[11]:


X = data.drop({'brand', '제품'}, axis=1)


# ## Data Preprocessing

# In[11]:


plt.figure(figsize=(10, 10))
f_corr = pd.DataFrame(data).corr()
sns.heatmap(f_corr, cmap='coolwarm', annot=True, annot_kws = {"size" : 7})
plt.rcParams['font.family'] = 'AppleGothic'
plt.show()
#plt.savefig('./heatmap.png', dpi=300)


# In[12]:


data.groupby('10대').sum()['Oily']


# In[17]:


data.groupby('brand').sum()['freq']


# In[20]:


df = pd.DataFrame(data.groupby('brand').mean()['평균값'])


# In[21]:


df


# In[34]:


plt.figure(figsize=(10, 6))
sns.lineplot(x='brand',y='평균값', data=df, palette='Accent')
plt.rcParams['font.family'] = 'AppleGothic'
plt.show()


# In[38]:


sns.countplot(x='brand', data=data, hue='freq', palette='Accent')


# In[43]:


y = data['freq'] # target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)
X_train


# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# In[35]:


from sklearn.model_selection import KFold, GridSearchCV 


# ## SVM

# In[44]:


kfold = KFold(n_splits=3, shuffle=True, random_state=14)

# SVM
param_range = [0.001, 0.01, 0.1, 0, 1, 10, 100]

param_grid = [{'svc__C': param_range,
               'svc__kernel' : ['linear']},
             {'svc__C': param_range, 
              'svc__gamma': param_range,
              'svc__kernel' : ['rbf']}]

pipe = make_pipeline(StandardScaler(),
                SVC(random_state=1))

grid_model = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=kfold,)

grid_model.fit(X_train, y_train)


# In[45]:


print('교차검증 점수: ', grid_model.best_score_)
print('최적의 하이퍼 파라미터 조합: ', grid_model.best_params_)
print('학습 평가: ', grid_model.score(X_train, y_train))
print('테스트 평가: ', grid_model.score(X_test, y_test))


# In[46]:


# refit with best hyperparameters
base_model = grid_model.best_estimator_
base_model.fit(X_train, y_train) 
base_model.score(X_test, y_test)


# In[47]:


from sklearn.metrics import classification_report


# In[50]:


svc_lin = SVC(C=10, gamma=0.01, kernel='linear')


# In[51]:


svc_lin.fit(X_train, y_train)
y_pred_lin = svc_lin.predict(X_test)


# In[52]:


print(classification_report(y_test,y_pred_lin))


# In[ ]:




