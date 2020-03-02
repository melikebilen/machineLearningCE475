import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]


X_test_final=df[['x1','x2','x3']][100:]

mse_lin=[]
r2_lin=[]


for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    pre = model.predict(X_test)
    mse_lin.append(mean_squared_error(y_test,pre))
    r2_lin.append(r2_score(y_test,pre))


print('MSE:',np.mean(mse_lin))
print('r squared',np.mean(r2_lin))
