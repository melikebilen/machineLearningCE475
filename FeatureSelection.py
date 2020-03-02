import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2


df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.
X= df[['x1','x2','x3','x4','x5','x6']][0:100]
y= df['Y'][0:100]

X_test_final=df[['x1','x2','x3','x4','x5','x6']][100:]

train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3','x4','x5','x6']][100:]

print(df.shape)

sel=SelectFromModel(LinearRegression())
sel.fit(train_X,train_y)
print(sel.get_support())
print(sel.estimator_.coef_)
features= train_X.columns[sel.get_support()]
print(features)

x_train_reg=sel.transform(train_X)
x_test_reg=sel.transform(test_X)
























#Teker teker aradaki ilişkiler
#sns.jointplot(X['x6'],y)
#plt.show()