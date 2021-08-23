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
import os
import re

# cd "C:\Users\82105\Desktop\SKKU\origins"

data = pd.read_csv('origins.csv', encoding = 'utf-8')

data.head()

data["age"] = data["age"].str.replace(pat=r'[^\w]', repl=r'', regex=True)

data["skintype"] = data["skintype"].str.replace(pat=r'[^\w]', repl=r'', regex=True)

data.head()

data["review"] = data["review"].str.replace(pat=r'[^\w]', repl=r' ', regex=True) # 특수문자 제거

data.head(10)

#def cleansing(text):
for text in data["review"]:
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

data["review"] = data["review"].str.replace(pat='([ㄱ-ㅎㅏ-ㅣ])+', repl=r' ', regex=True) # 한글 자음, 모음 제거

data.head(10)

pattern = re.compile(r'\s+') # 이중 공백 제거
data["review"] = data["review"].str.replace(pat=pattern, repl=' ', regex=True)
