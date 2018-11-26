#-*- coding:UTF-8 -*-
#@Time:  
#@Author TianYuhan
#@File: .py
#@Software: PyCharm
import pandas as pd
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\Users\Tian\Desktop\Data\LogisticRegression.csv',encoding='UTF-8')
print data.head(5)
