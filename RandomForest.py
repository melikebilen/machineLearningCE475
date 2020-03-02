from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
#Random forest is an extention of bagged decision trees

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3']][100:]
#train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)

#regressor = RandomForestRegressor(n_estimators = 6,max_depth=3,bootstrap=True) #rf = RandomForestRegressor(n_estimators=100, criterion='mse',random_state=1, n_jobs=-1)
#regressor.fit(train_X,train_y)

#choosing the best n_estimators

mse_different_est42=[]
""""
for i in range(10):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    reg = RandomForestRegressor(n_estimators=i+1)
    reg.fit(train_X,train_y)
    t_pre=reg.predict(test_X)
    mse_choosing_nEstimator=mean_squared_error(test_y,t_pre)
    mse_different_est42.append(mse_choosing_nEstimator)
print('MSE OF THE DIFFERENT VALUES random state 42',mse_different_est42)
"""

for i in range(15):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        reg = RandomForestRegressor(n_estimators=i + 1)
        reg.fit(train_X, train_y)
        t_pre = reg.predict(test_X)
        mse_choosing_nEstimator = mean_squared_error(test_y, t_pre)
        mse_different_est42.append(mse_choosing_nEstimator)

print('MSE OF THE DIFFERENT VALUES random state', mse_different_est42)




bir=[]
iki=[]
üç=[]
dört=[]
beş=[]
altı=[]
yedi=[]
sekiz=[]
dokuz=[]
on=[]



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
predictionsArray=[]
error=[]
mse=[]


for i in range(20):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    regressor = RandomForestRegressor(n_estimators=10,max_depth=5,max_features='auto',bootstrap=True)
    regressor.fit(train_X,train_y)

    test_prediction= regressor.predict(test_X)
    error.append(r2_score(test_y,test_prediction))
    mse.append(mean_squared_error(test_y,test_prediction))

    pre=regressor.predict(X_test_final)
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


print('mse:',np.mean(mse))
print('r2 test error of 20 trials:',np.mean(error))



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


