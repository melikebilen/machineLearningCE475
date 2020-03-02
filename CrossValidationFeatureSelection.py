import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import SCORERS,mean_squared_error

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.

X= df[['x1','x2','x3','x4','x5','x6']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3','x4','x5','x6']][100:]

lm=LinearRegression()

score= cross_val_score(lm,X,y,cv=10,scoring='neg_mean_squared_error')

#fix the signs
#mse_score=-score
#rmse
#rmse_score=np.sqrt(mse_score)
#print('rmse:',rmse_score.mean())


#####Excluding x1
X1siz= df[['x2','x3','x4','x5']][0:100]
print('x2,x3,x4,x5,x6',(cross_val_score(lm,X1siz,y,cv=15)).mean())

#####Excluding x2 and x6
X26sız= df[['x1','x3','x4','x5']][0:100]
print('x1,x3,x4,x5',(cross_val_score(lm,X26sız,y,cv=15)).mean())


#####Excluding x3
X3suz= df[['x1','x2','x4','x5']][0:100]
print('x1,x2,x4,x5',(cross_val_score(lm,X3suz,y,cv=15)).mean())


####excluding 4
X4suz= df[['x1','x2','x3','x5']][0:100]
print('x1,x2,x3,x5',(cross_val_score(lm,X4suz,y,cv=20)).mean())


####excluding 5
X5suz= df[['x1','x2','x3','x4']][0:100]
print('x1,x2,x3,x4',(cross_val_score(lm,X5suz,y,cv=15)).mean())

#####Excluding x4 x5 x6
X123lı= df[['x1','x2','x3']][0:100]
print('x1,x2,x3',(cross_val_score(lm,X123lı,y,cv=15)).mean())

#####Excluding  x2,x4,x5,x6
X13lı= df[['x1','x3']][0:100]
print('X1,x3',(cross_val_score(lm,X13lı,y,cv=15)).mean())


#####Excluding x1 x2 x3
X456lı= df[['x1','x4','x5',]][0:100]
print('x4,x5',(cross_val_score(lm,X456lı,y,cv=15)).mean())


#####Excluding x1 x2 x3 x6
X45li= df[['x4','x5']][0:100]
print('x4,x5',(cross_val_score(lm,X45li,y,cv=15)).mean())

