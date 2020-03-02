import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')

X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

X_test_final=df[['x1','x2','x3','x4','x5']][100:]
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


#Linear Regression Obj
lm=LinearRegression()

#Random Forest Obj
reg=RandomForestRegressor()

#Polynomial Obj
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
pre=pol_reg.predict(poly_reg.fit_transform(X))

#decision tree objesi
tree= tree.DecisionTreeRegressor()

#lasso


lasso = linear_model.Lasso(alpha=30)

def get_score(model,train_X,test_X,train_y,test_y):
    model.fit(train_X,train_y)
    return model.score(test_X,test_y)

scores_l=[]
scores_poly=[]
scores_randForest=[]
scores_randEst20=[]
scores_gradient=[]
scores_tree=[]
scores_lasso=[]
#print(cross_val_score(lm,X,y,cv=5))
#print(cross_val_score(reg,X,y,cv=5))



scores_poly=cross_val_score(pol_reg,poly_reg.fit_transform(X),y,cv=5)
scores_l=cross_val_score(lm,X,y,cv=5)
scores_randForest=cross_val_score(reg,X,y,cv=5)
scores_randEst20=cross_val_score(RandomForestRegressor(n_estimators=20,criterion='mse',bootstrap=True),X,y,cv=5)
scores_randEst15=cross_val_score(RandomForestRegressor(n_estimators=15,criterion='mse',bootstrap=True),X,y,cv=5)
scores_gradient=cross_val_score(ensemble.GradientBoostingRegressor(),X,y,cv=5)
scores_tree=cross_val_score(tree,X,y,cv=5)
scores_lasso=(cross_val_score(lasso, X, y, cv=5))

print('linear cross val mean:',scores_l.mean())
print('Lasso cross val mean:',scores_lasso.mean())
print('poly val mean default:',scores_poly.mean())
print('tree val mean default:',scores_tree.mean())
print('random forest cross val mean (estimator 10):',scores_randForest.mean())
print('random forest cross val mean (estimator 15):',scores_randEst15.mean())
print('random forest cross val mean (estimator 20):',scores_randEst20.mean())
print('gradient boosting val mean default:',scores_gradient.mean())

