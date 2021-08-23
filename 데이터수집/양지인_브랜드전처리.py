#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import re


# In[3]:


cd '/Users/stella/Downloads'


# In[12]:


data = pd.read_csv('roundlab.csv')


# In[13]:


data.head()


# In[14]:


data.isnull().sum(axis=0)


# In[77]:


data["age"] = data["age"].str.replace(pat=r'[^\w]', repl=r'', regex=True)


# In[78]:


data["skintype"] = data["skintype"].str.replace(pat=r'[^\w]', repl=r'', regex=True)


# In[79]:


data.head()


# In[80]:


data["review"] = data["review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True) # 특수문자 제거


# In[81]:


data.head(10)


# In[36]:


#def cleansing(text):
    pattern = '(\[a-zA-Z0-9\_.+-\]+@\[a-zA-Z0-9-\]+.\[a-zA-Z0-9-.\]+)' # 이메일 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:\da-fA-F]{2}))+' # url 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = '([ㄱ-ㅎㅏ-ㅣ])+' # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = '<[^>]*>' #html tag 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = '[\r|\n]' # \r, \n 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = '[^\w\s]' # 특수기호 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
    
    pattern = re.compile(r'\s+') # 이중 공백 제거
    text = re.sub(pattern=pattern, repl=' ', string=text)
  #  return(text)


# In[82]:


data["review"] = data["review"].str.replace(pat='([ㄱ-ㅎㅏ-ㅣ])+', repl=r' ', regex=True) # 한글 자음, 모음 제거


# In[83]:


data.head(10)


# In[84]:


pattern = re.compile(r'\s+') # 이중 공백 제거
data["review"] = data["review"].str.replace(pat=pattern, repl=' ', regex=True)


# In[85]:


data.to_csv('abib_cleaned.csv')


# In[ ]:




