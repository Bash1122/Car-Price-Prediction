import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns; 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

#load dataset
df = pd.read_csv('vehicle_price.csv')
print(df)

#data types of cloums 
df.dtypes

#shape
df.shape

describe=df.describe()
print(describe)

#Find the missing values
missing_values = df.isna()
print(missing_values)

#Dropping the null values
print(df.isnull().sum())
df = df.dropna()

#After droping the null values
missing_values = df.isna()
print(missing_values)

#fuel types 
df['fuel'].value_counts()

df['fuel'].nunique()

#brand
df['brand'].nunique()

#body
df['body'].nunique()

#Rename
df = df.rename(columns = {"price$":"price"}) 

df.isnull().sum()

#shape
df.shape

#Histogram for fuel
plt.hist(df['fuel'],bins =int(100/5),color = 'black')
plt.title('Histogram of fuel type')
plt.xlabel('fuel')
plt.show()

#Histogram for price
plt.hist(df['price'],bins =int(100/5),color = 'red')
plt.title('Histogram of price')
plt.xlabel('price')
plt.show()

#Histogram for car_mileage
plt.hist(df['car_mileage'],bins =int(100/5),color = 'blue')
plt.title('Histogram of car_mileage')
plt.xlabel('car_mileage')
plt.show()

#Histogram for brand
plt.hist(df['brand'],bins =int(100/5),color = 'red')
plt.title('Histogram of brand')
plt.xlabel('brand')
plt.show()


#price vs brand graph
fig=plt.figure()
fig= plt.figure(figsize=(16,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((df['brand']),df['price'])
plt.title('price vs brand ')
plt.xticks(rotation=90)
plt.xlable(df['brand'])
plt.ylable(df['price'])
plt.show()

#brand vs car_millage
fig=plt.figure()
fig= plt.figure(figsize=(16,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((df['brand']),df['car_mileage'])
plt.title('brand vs car_millage')
plt.xticks(rotation=90)
plt.xlable(df['brand'])
plt.ylable(df['car_mileage'])
plt.show()

#Coverting strings in to numbers
le = preprocessing.LabelEncoder()
le.fit(df['brand'])
df['brand']=le.transform(df['brand'])

le = preprocessing.LabelEncoder()
le.fit(df['body'])
df['body']=le.transform(df['body'])
#print(data1)
le = preprocessing.LabelEncoder()
le.fit(df['fuel'])
df['fuel']=le.transform(df['fuel'])

le = preprocessing.LabelEncoder()
le.fit(df['transmission'])
df['transmission']=le.transform(df['transmission'])

#data slicing
X = df.iloc[:, 0:6]
y = df.iloc[:, 7]


#Splitting data into training and testing set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_test)
print(y_test)

#Create a Model 

#Linear Regression

model_1 = LinearRegression()
model_1.fit(X_train,y_train)
l_reg = model_1.score(X_test, y_test)
print(l_reg)
y_pred=model_1.predict(X_test)
print(y_pred)
r2_score(y_test, y_pred)


#Linear regression with cross-validation
from sklearn.model_selection import cross_val_score
model_1 = LinearRegression()
scores = cross_val_score(model_1, X, y, cv=10)
print("Cross-validation scores: {}".format(scores))
m1 = "{:.2f}".format(scores.mean())
std1 = "{:.2f}".format(scores.std())
print(m1)
print(std1)
from sklearn.metrics import mean_absolute_error
regression_error=mean_absolute_error(y_test, y_pred)
print(regression_error)


fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(y_pred))
plt.title('y_test and y_pred after Linear reg')
plt.xticks(rotation=90)
plt.show()


# Decision tree Regression:
from sklearn.tree import DecisionTreeRegressor
model_2=DecisionTreeRegressor()
model_2.fit(X_train,y_train)
decc=model_2.score(X_test,y_test)
print(decc)
predict = model_2.predict(X_test)
print(predict)
from sklearn.metrics import mean_absolute_error
Ds_error=mean_absolute_error(y_test,predict)
print(Ds_error)


#Decision tree Regression with cross-validation:-
from sklearn.model_selection import cross_val_score
model_2 = DecisionTreeRegressor()
scores = cross_val_score(model_2, X, y, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
m2 = "{:.2f}".format(scores.mean())
std2 = "{:.2f}".format(scores.std())
print(m2)
print(std2)
from sklearn.metrics import mean_absolute_error
Ds_error = mean_absolute_error(y_test, predict)
print(Ds_error)


#Model 3 Random Forest Regression with train and test
model_3 = RandomForestRegressor()
model_3.fit(X_train, y_train)
random=model_3.score(X_test,y_test)
print(random)
pred=model_3.predict(X_test)
print(pred)
r2_score(y_test, pred)

from sklearn.metrics import mean_absolute_error
rendomf_error= mean_absolute_error(y_test, pred)
print(rendomf_error)


fig=plt.figure()
fig= plt.figure(figsize=(6,6))
ax=fig.add_subplot(1,1,1)
ax.scatter((y_test),(pred))
plt.title('y_test and pred after random f')
plt.xticks(rotation=90)
plt.show()

#Model 3 Random Forest Regression with cross-validation
model_3 = RandomForestRegressor()
scores = cross_val_score(model_3, X, y, cv=10)
print("Cross-validation scores: {}".format(scores))
#print("Average cross-validation score: {:.2f}".format(scores.mean()))
m3 = "{:.2f}".format(scores.mean())
std3 = "{:.2f}".format(scores.std())
print(m3)
print(std3)
from sklearn.metrics import mean_absolute_error
randomf_error = mean_absolute_error(y_test, predict)
print(randomf_error)


#Model Tunning

from sklearn.model_selection import GridSearchCV
no_of_test =[100]
params_dict = {'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto","sqrt","log2"]}
model_3=GridSearchCV(estimator = RandomForestRegressor(),param_grid=params_dict,scoring='r2')
model_3.fit(X_train, y_train)
print('Score: %.4f' %model_3.score(X_test, y_test))
pred = model_3.predict(X_test)
r2= r2_score(y_test, pred)
print('r2 :%0.2f' %r2)
print('R2     : %0.2f ' % r2)



#Model Comparison
print('                       ##########  Model Comparison  ##########  ')
models=pd.DataFrame({
                     'Model':['linearRegession','Decision Tree','RandomForestReression'],
                     'Score':[l_reg,decc,random],
                    'Cross-validation scores':[m1,m2,m3],
                     'mean_absolute_error':[regression_error,Ds_error,rendomf_error]})
print(models)
result=pd.DataFrame({'Mean':[m1,m2,m3],'STD':[std1,std2,std3]})
Model=['linearRegession','Decision Tree','RandomForestReression']
print(result)
result=result.astype(float)
   

#Boxplot algorithm comparison

fig = plt.figure()
fig= plt.figure(figsize=(10,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(result)
ax.set_xticklabels(Model)
plt.show()




