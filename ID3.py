#-*- coding:UTF-8 -*-
#@Time:  
#@Author TianYuhan
#@File: .py
#@Software: PyCharm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC,export_graphviz

data = pd.read_csv('C:\Users\Tian\Desktop\Data\\titanic_data.csv',encoding='UTF-8')
data.drop(['PassengerId'],axis=1,inplace=True)
data.loc[data['Sex']=='male','Sex'] = 1
data.loc[data['Sex']=='female','Sex'] = 0
data.fillna(int(data.Age.mean()),inplace=True)
#print data.head(5)
x = data.iloc[:,1:3]
y = data.iloc[:,0]

dtc = DTC(criterion='entropy')
dtc.fit(x,y)
print '输出准确率:',dtc.score(x,y)

with open('C:\Users\Tian\Desktop\Data\\tree.dot','w') as f:
    f = export_graphviz(dtc,feature_names=x.columns,out_file=f)
