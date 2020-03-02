from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.
X= df[['x1','x2','x3','x4','x5']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3','x4','x5']][100:]


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)

linear= LinearRegression()
linear.fit(train_X,train_y)
linpre=linear.predict(test_X)

lasso = linear_model.Lasso(alpha=10)
lasso.fit(train_X,train_y)
laspre=lasso.predict(test_X)

ridge = linear_model.Ridge(alpha=10)
ridge.fit(train_X,train_y)
ridpre= ridge.predict(test_X)

lin=cross_val_score(linear,X,y,cv=10)
las=cross_val_score(lasso,X,y,cv=10)
rid=cross_val_score(ridge,X,y,cv=10)

print('linear cv:',lin.mean())
print('lasso cv:',las.mean())
print('Ridge cv:',rid.mean())



print('linear r2:',r2_score(test_y,linpre))
print('lasso r2:',r2_score(test_y,laspre))
print('Ridge r2:',r2_score(test_y,ridpre))



