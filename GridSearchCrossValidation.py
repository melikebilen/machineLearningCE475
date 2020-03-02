from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

#Random forest is an extention of bagged decision trees

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]
X_test_final=df[['x1','x2','x3']][100:]
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=101)


param_grid={'n_estimators':[1,2,3,4,5,6,7,8,9,10],'max_depth':[1,2,3,4,5,6,7],'bootstrap':[True,False]}
rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid =param_grid,cv = 30)
grid_search=grid_search.fit(train_X, train_y)


best_grid = grid_search.best_estimator_
best_prediction=best_grid.predict(test_X)
best_training=best_grid.predict(train_X)
print(grid_search.best_params_)

base = RandomForestRegressor(n_estimators = 3)
base.fit(train_X, train_y)
base_training= base.predict(train_X)
predictions = base.predict(test_X)


print('base mse: ',mean_squared_error(test_y,predictions))
print('base r2',r2_score(test_y,predictions))
print('base r2 training error:',r2_score(train_y,base_training))

print('best mse: ',mean_squared_error(test_y,best_prediction))
print('best r2',r2_score(test_y,best_prediction))
print('best r2 training error:',r2_score(train_y,best_training))


