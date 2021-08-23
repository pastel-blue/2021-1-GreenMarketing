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

os.chdir('\\Users\\82105\\Desktop\\새 폴더')
os.getcwd()

# +
# 각 브랜드 성분, 선호도 점수 파일 불러오기

abib = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/abib_prefer.csv", engine='python') 
aveda = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/aveda_prefer.csv", engine='python') 
innisfree = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/innisfree_prefer.csv", engine='python') 
kiehl = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/kiehl_prefer.csv", engine='python') 
lush = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/lush_prefer.csv", engine='python') 
melixir = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/melixir_prefer.csv", engine='python') 
origins = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/origins_prefer.csv", engine='python') 
roundlab = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/roundlab_prefer.csv", engine='python') 
thebodyshop = pd.read_csv("C:/Users/82105/Desktop/SKKU/prefer/data_prefer/thebodyshop_prefer.csv", engine='python')

# +
# 각각의 개별 브랜드 파일을 하나로 합칠 때 해당 브랜드가 어디인지 알 수 있도록 데이터프레임 맨 앞에 브랜드명 추가

abib.insert(0, 'brand', "abib")
innisfree.insert(0, 'brand', "innisfree")
kiehl.insert(0, 'brand', "kiehl")
lush.insert(0, 'brand', "lush")
melixir.insert(0, 'brand', "melixir")
origins.insert(0, 'brand', "origins")
roundlab.insert(0, 'brand', "round lab")
aveda.insert(0, 'brand', "aveda")
thebodyshop.insert(0, 'brand', "the body shop")

# +
# 다른 브랜드들과 col 제목이 일부 다른 the body shop 데이터프레임의 col명 수정

thebodyshop = thebodyshop.rename(columns = {"teens":"10대"})
thebodyshop = thebodyshop.rename(columns = {"twenties":"20대"})
thebodyshop = thebodyshop.rename(columns = {"thirties":"30대"})
thebodyshop = thebodyshop.rename(columns = {"overforties":"40대 이상"})

# +
# 브랜드들 개별 데이터프레임 하나로 합치기

total_data = abib.append([aveda, innisfree, kiehl, lush, 
                          melixir, origins, roundlab, thebodyshop])

total_data = total_data.reset_index(drop=True)
# -

total_data

# rating (선호도) col에 존재하는 결측치는 모두 4, 5점에 해당하는 긍정 점수임. 수기로 선호도를 입력하는 과정에서 막판에 4, 5점은 비워놓고 부정 점수만 채우느라 결측값이 존재함. 따라서 결측값을 긍정에 해당하는 5점으로 채워주겠음 (아래 코드)

# +
# rating 열 결측치 긍정점수 (5.0)으로 채우기

total_data['rating'] = total_data['rating'].replace(np.nan, 5.0)

# +
# 필요없는 col (의미없는 인덱스열) drop

total_data = total_data.drop(["Column1"], axis=1)
# -

total_data
