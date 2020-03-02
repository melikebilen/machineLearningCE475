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
#Random forest is an extention of bagged decision trees

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3']][100:]
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 15)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 50, num = 3)]
max_depth.append(None)
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}
print(random_grid)




# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 50 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 6, verbose=2)
# Fit the random search model
rf_random.fit(X,y)
print('--------------------------------------------------')
print(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print(r2_score(test_features,predictions))


base_model = RandomForestRegressor(n_estimators = 10)
base_model.fit(train_X,train_y)
print('base accuracy')
evaluate(base_model,test_X,test_y)


best_random = rf_random.best_estimator_
print('best accuracy')
evaluate(best_random,test_X, test_y)











