import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
import pydot

df = pd.read_csv('CE 475 Fall 2019 Project Data - Data.csv')
#Modelimi oluşturmak için kullanacağım data bu.
X= df[['x1','x2','x3']][0:100]
y= df['Y'][0:100]

#Prediction elde edeceğim x değerleri
X_test_final=df[['x1','x2','x3']][100:]


depthTwoMse=[]
depthThreeMse=[]
depthFourMse=[]
depthFiveMse=[]
r2_3=[]
r2_5=[]

for i in range(10):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    model = tree.DecisionTreeRegressor(max_depth=2)
    model.fit(train_X, train_y)
    p2= model.predict(test_X)
    depthTwoMse.append(mean_squared_error(test_y,p2))


    model3 = tree.DecisionTreeRegressor(max_depth=3)
    model3.fit(train_X, train_y)
    p3=model3.predict(test_X)
    depthThreeMse.append(mean_squared_error(test_y,p3))
    r2_3.append(r2_score(test_y, p3))

    model4 = tree.DecisionTreeRegressor(max_depth=4)
    model4.fit(train_X, train_y)
    p4=model4.predict(test_X)
    depthFourMse.append(mean_squared_error(test_y,p4))

    model5 = tree.DecisionTreeRegressor(max_depth=5)
    model5.fit(train_X, train_y)
    p5 = model5.predict(test_X)
    depthFiveMse.append(mean_squared_error(test_y, p5))
    r2_5.append(r2_score(test_y,p5))



print('MSE using depth 2',np.mean(depthTwoMse))
print('R2 score (depth 3):',np.mean(r2_3))
print('MSE using (depth 3):',np.mean(depthThreeMse))
print('MSE using depth 4',np.mean(depthFourMse))
print('MSE using (depth 5):',np.mean(depthFiveMse))
print('R2 score (depth 5):',np.mean(r2_5))







train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=101)
model= tree.DecisionTreeRegressor(max_depth=2)
model.fit(train_X,train_y)


train_pre= model.predict(train_X)
first_pre= model.predict(test_X)

print('traning error:',r2_score(train_y,train_pre))
print('R2 score:',r2_score(test_y,first_pre))
print('mean squared error:',mean_squared_error(test_y,first_pre))
print('Cross validation score',cross_val_score(model,X,y,cv=10).mean())



print('AND OUR FINAL PREDICTIONS')
predictions= model.predict(X_test_final)
print(predictions)



plt.scatter(test_y,first_pre)
plt.xlabel('true y')
plt.ylabel('predictions using test values')

