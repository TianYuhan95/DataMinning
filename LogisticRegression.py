#-*- coding:UTF-8 -*-
#@Time:  
#@Author TianYuhan
#@File: .py
#@Software: PyCharm
import pandas as pd
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\Users\Tian\Desktop\Data\LogisticRegression.csv',encoding='UTF-8')
#print data.head(5)
data_dum = pd.get_dummies(data,prefix='rank',columns=['rank'],drop_first=True)
#print data_dum.tail(5)
x_train,x_test,y_train,y_test = train_test_split(data_dum.ix[:,1:],data_dum.ix[:,0],test_size=.1,random_state=520)
lr = LogisticRegression()
lr.fit(x_train,y_train)
print '逻辑回归的准确率为:{0:.2f}%'.format(lr.score(x_test,y_test)*100)