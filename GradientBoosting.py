import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

X_test_final=df[['x1','x2','x3',]][100:]



""""
decidingEstimator=[]

for i in range(15):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=101)
    GBR = GradientBoostingRegressor(n_estimators=i+1,learning_rate=0.1)
    GBR.fit(train_X,train_y)
    prediction=GBR.predict(test_X)
    decidingEstimator.append(mean_absolute_error(test_y,prediction))
    
print('Estimators,',decidingEstimator)
"""""

"""""
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
search_grid={'n_estimators':[8,9,10,11,12,13,14,15,16],'learning_rate':[.001,0.01,.1],'max_depth':[5,6],'max_features':['auto','sqrt']}
search=GridSearchCV(estimator=GBR,param_grid=search_grid,n_jobs=1,cv=20)
search=search.fit(train_X,train_y)

#search.best_params_
print('best parameter:',search.best_params_)
print('estimator',search.best_estimator_)
best_GBR=search.best_estimator_
best_prediction=best_GBR.predict(test_X)
print('mse best:',mean_squared_error(test_y,best_prediction))
print('R2 best:',r2_score(test_y,best_prediction))

"""""

r2_gbr=[]
mse_gbr=[]

p0=[]
p1=[]
p2=[]
p3=[]
p4=[]
p5=[]
p6=[]
p7=[]
p8=[]
p9=[]
p10=[]
p11=[]
p12=[]
p13=[]
p14=[]
p15=[]
p16=[]
p17=[]
p18=[]
p19=[]
p20=[]


GBR=GradientBoostingRegressor(n_estimators=16,learning_rate=0.1,max_depth=5,max_features='auto')
for i in range(20):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    """"
    search_grid = {'n_estimators': [8, 9, 10, 11, 12, 13, 14, 15, 16], 'learning_rate': [.001, 0.01, .1],
                   'max_depth': [5, 6, 7], 'max_features': ['auto', 'sqrt']}
    search = GridSearchCV(estimator=GBR, param_grid=search_grid, n_jobs=1, cv=20)
    search = search.fit(train_X, train_y)
    best_GBR = search.best_estimator_
   # print('best parameters',search.best_params_)
    best_prediction = best_GBR.predict(test_X)
    """""
    GBR.fit(train_X,train_y)
    best_prediction = GBR.predict(test_X)
    mse_gbr.append(mean_squared_error(test_y, best_prediction))
    r2_gbr.append(r2_score(test_y, best_prediction))


    pre=GBR.predict(X_test_final)
    p0.append(pre[0])
    p1.append(pre[1])
    p2.append(pre[2])
    p3.append(pre[3])
    p4.append(pre[4])
    p5.append(pre[5])
    p6.append(pre[6])
    p7.append(pre[7])
    p8.append(pre[8])
    p9.append(pre[9])
    p10.append(pre[10])
    p11.append(pre[11])
    p12.append(pre[12])
    p13.append(pre[13])
    p14.append(pre[14])
    p15.append(pre[15])
    p16.append(pre[16])
    p17.append(pre[17])
    p18.append(pre[18])
    p19.append(pre[19])



print('Gradient boosting r2 val:',np.mean(r2_gbr))
print('Gradient boosting mse val:',np.mean(mse_gbr))



print(np.mean(p0))
print(np.mean(p1))
print(np.mean(p2))
print(np.mean(p3))
print(np.mean(p4))
print(np.mean(p5))
print(np.mean(p6))
print(np.mean(p7))
print(np.mean(p8))
print(np.mean(p9))
print(np.mean(p10))
print(np.mean(p11))
print(np.mean(p12))
print(np.mean(p13))
print(np.mean(p14))
print(np.mean(p15))
print(np.mean(p16))
print(np.mean(p17))
print(np.mean(p18))
print(np.mean(p19))