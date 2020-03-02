from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#DECIDING WHICH DEGREE TO USE
df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')

X= df[['x1','x2','x3']][0:100].values
y= df['Y'][0:100].values

X_test_final=df[['x1','x2','x3']][100:]

mse_2=[]
mse_3=[]
mse_4=[]
mse_5=[]

r2_2=[]
r2_3=[]
r2_4=[]
r2_5=[]


for i in range(20):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    poly_reg2 = PolynomialFeatures(degree=2)
    X_poly2 = poly_reg2.fit_transform(train_X)
    pol_reg2 = LinearRegression()
    pol_reg2.fit(X_poly2, train_y)
    pre2 = pol_reg2.predict(poly_reg2.fit_transform(test_X))
    mse_2.append(mean_squared_error(test_y,pre2))
    r2_2.append(r2_score(test_y,pre2))

    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(train_X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, train_y)
    pre3 = pol_reg.predict(poly_reg.fit_transform(test_X))
    mse_3.append(mean_squared_error(test_y, pre3))
    r2_3.append(r2_score(test_y, pre3))


    poly_reg4 = PolynomialFeatures(degree=4)
    X_poly4 = poly_reg4.fit_transform(train_X)
    pol_reg4 = LinearRegression()
    pol_reg4.fit(X_poly4, train_y)
    pre4 = pol_reg4.predict(poly_reg4.fit_transform(test_X))
    mse_4.append(mean_squared_error(test_y, pre4))
    r2_4.append(r2_score(test_y, pre4))


    poly_reg5 = PolynomialFeatures(degree=5)
    X_poly5 = poly_reg5.fit_transform(train_X)
    pol_reg5 = LinearRegression()
    pol_reg5.fit(X_poly5, train_y)
    pre5 = pol_reg5.predict(poly_reg5.fit_transform(test_X))
    mse_5.append(mean_squared_error(test_y, pre5))
    r2_5.append(r2_score(test_y, pre5))




pre_final=pol_reg.predict(poly_reg.fit_transform(X_test_final))

print('mean squared error(degree 2):',np.mean(mse_2))
print('R2 score (degree 2):',np.mean(r2_2))

print('mean squared error(degree 3):',np.mean(mse_3))
print('R2 score(degree 3):',np.mean(r2_3))

print('test error degree 4:',np.mean(mse_4))
print('R2 score(degree 4):',np.mean(r2_4))

print('test error degree 5:',np.mean(mse_5))
print('R2 score(degree 5):',np.mean(r2_5))

