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

pip install selenium

# +
import time
import re 
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#각자 드라이버 위치 맞게 넣어주기
driver = webdriver.Chrome("/Users/geeee/Desktop/디스커버/chromedriver")
driver.implicitly_wait(2)
driver.get('https://www.glowpick.com/brand/ranking?id=186&gender=all&age=all&skin_type=all')

#body click
body = driver.find_element_by_tag_name("body") 
body.click()

#스크롤 다운
page = 1
elem = driver.find_element_by_tag_name("body") 

#스크롤 내리기(여기선 30번)
while page <= 30:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.3)
    page+=1

# +
data ={}
model_list = [] 
nickname_list= []
age_list = []
skintype_list = []
review_list = []


#제품 페이지간 이동(제품 100개)
for i in range(0,100):
    time.sleep(0.5)
    
    #제품 리스트 추출 
    thebodyshop_All = driver.find_elements_by_css_selector("div > div > div.product-item__info__details.details > div.details__labels > p.details__labels__name") 
    
    model = thebodyshop_All[i].text
    print(i,"번째 제품: ", model)
    
    #제품 페이지 들어가기
    thebodyshop_All[i].click()
    time.sleep(2)
    
    #스크롤 내리기
    body = driver.find_element_by_tag_name("body")
    body.click()
    
    page = 1
    while page <= 100:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
        page+=1
    
    
    #리뷰 개수 파악
    ages = driver.find_elements_by_css_selector("div > div > div > p > span.info > span.txt")
    num = (len(ages))
    print("리뷰 개수: ", num)
    
    #리뷰 하나씩 저장 
    for j in range(0, num):
        #제품명 저장 
        model_list.append(model)
        
        age = ages[j].text
        skintype = age
        
        #전처리 여기서 작동 안됨
        age = re.findall("\d+", age)
        #age = re.sub("[\[\]\']","",age)
        skintype  = re.findall("[가-힣]{3}", skintype)
        #age = re.sub("[\[\]\']","",age)
        
        #리뷰부분 저장
        reviews = driver.find_elements_by_class_name("review")
        review = reviews[j].text
        #review = review.sub(r'[^\w]', "", review)
        
        #리스트에 저장
        age_list.append(age)
        skintype_list.append(skintype)
        review_list.append(review)
        
    driver.back()
    time.sleep(0.5)
   
# -

data = {'제품':model_list, 'age':age_list, 'skintype':skintype_list, 'review':review_list}
df = pd.DataFrame(data)
df

df.to_csv("이니스프리.csv")


