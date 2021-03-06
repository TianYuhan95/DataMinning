#-*- coding:UTF-8 -*-
#@Time:  
#@Author TianYuhan
#@File: .py
#@Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
#print(boston.keys())
#print(boston.feature_names)
#print(boston.DESCR)

x = boston.data[:,np.newaxis,5]
print(x)
y = boston.target
lm = LinearRegression()
lm.fit(x,y)
print(u'方程的确定性系数(R^2):%.2f'%lm.score(x,y))

plt.scatter(x,y,color='green')
plt.plot(x,lm.predict(x),color='blue',linewidth=3)
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Housing Price')
plt.title('2D Demo of Line Regression')
plt.show()
