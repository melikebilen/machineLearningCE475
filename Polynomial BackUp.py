from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
X= df[['x1','x2','x3']][0:100].values
y= df['Y'][0:100].values
X_test_final=df[['x1','x2','x3']][100:]

""""
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(train_X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, train_y)

pre_final=pol_reg.predict(poly_reg.fit_transform(X_test_final))
pre_y=pol_reg.predict(poly_reg.fit_transform(test_X))
print('R2 value',r2_score(test_y,pre_y))
print('FINAL PREDICTIONS',pre_final)
"""


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
predictionsArray=[]
error=[]
mse=[]
mae=[]

for i in range(20):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(train_X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, train_y)

    preY = pol_reg.predict(poly_reg.fit_transform(test_X))

    error.append(r2_score(test_y, preY))
    mse.append(mean_squared_error(test_y,preY))
    mae.append(mean_absolute_error(test_y,preY))

    pre = pol_reg.predict(poly_reg.fit_transform(X_test_final))
    predictionsArray.append(pre)
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

#scores_poly=cross_val_score(pol_reg,poly_reg.fit_transform(X),y,cv=7)
#print('cross val',scores_poly.mean())
print('r2 error of 20 iterations',np.mean(error))
print('mse error of 20 iterations',np.mean(mse))
print('mae error of 20 iterations',np.mean(mae))
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

