from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.
X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3']][100:]

lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=5).mean())

mse_lin=[]
r2_lin=[]


for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = linear_model.Lasso()
    model.fit(X_train, y_train)
    pre = model.predict(X_test)
    mse_lin.append(mean_squared_error(y_test,pre))
    r2_lin.append(r2_score(y_test,pre))


print('Lasso MSE:',np.mean(mse_lin))
print('Lasso r squared',np.mean(r2_lin))