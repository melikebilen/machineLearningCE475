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

#Random forest is an extention of bagged decision trees

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]
X_test_final=df[['x1','x2','x3']][100:]
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.1)

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 15)]
max_depth = [int(x) for x in np.linspace(1, 50, num = 3)]
max_depth.append(None)
bootstrap = [True, False]
max_features = ['auto', 'sqrt']
min_samples_split = [2,3,5]
min_samples_leaf = [1,2,3,4]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'max_features': max_features,
               'min_samples_leaf': min_samples_leaf,
               'min_samples_split': min_samples_split,
               'bootstrap': bootstrap
                }


rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 20)
rf_random.fit(train_X, train_y)
best_random = rf_random.best_estimator_

print(rf_random.best_params_)
best_prediction=best_random.predict(test_X)


base = RandomForestRegressor(n_estimators=4)
base.fit(train_X, train_y)
predictions = base.predict(test_X)


print('base mse: ',mean_squared_error(test_y,predictions))
print('base r2',r2_score(test_y,predictions))

print('best mse: ',mean_squared_error(test_y,best_prediction))
print('best r2',r2_score(test_y,best_prediction))


#{'n_estimators': 13, 'max_features': 'auto', 'max_depth': 50, 'bootstrap': True}