# Assignment Neural Networks on forestfires dataset using SKLEARN MLP

# Business Problem: To classify (or predict) the size (or area) of forest fires

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
pd.set_option('display.max_columns' , 50)

fire = pd.read_csv('E:/Neural Networks/forestfires.csv')
fire.head(3)
fire.columns
fire.describe()
fire.dtypes

# month and day, have already been converted to dummy variables, hence removing these variables
#in size_category, area<6 is considered small and area>=6 is big, hence remove area
fire.drop(['month','day','area'],  axis=1, inplace=True)
fire.shape # 517 and 28 var

# size_category is the dependant variable
fire.size_category.value_counts() # small  378 , large  139


# separating predictors and target
X = fire.drop(['size_category'],axis=1)
Y = fire['size_category']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.7, random_state=123)

# standardizing predictors
scaler = StandardScaler()
scaler.fit(Xtrain[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] )
Xtrain[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] = scaler.transform(Xtrain[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']])
Xtrain.describe()

Xtest[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] = scaler.transform(Xtest[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']])
Xtest.describe()

### running the model
mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,50,50),random_state=123)
mlp_classifier
mlp_classifier.fit(Xtrain, Ytrain)

## predicting
pred_train = mlp_classifier.predict(Xtrain)
pred_test = mlp_classifier.predict(Xtest)

######## evaluating training accuracy
Ytrain.value_counts()
'''
small    121
large     34 '''

pd.crosstab(Ytrain ,pred_train)
'''
col_0          large  small
size_category              
large             12     22
small              1    120 '''

metrics.accuracy_score(Ytrain, pred_train) # 0.85161
print(metrics.classification_report(Ytrain, pred_train))
# high proportion of small forest fires have been correctly classified but 
# very less proportion of large forest fires have been correctly classified

# evaluating training accuracy
Ytest.value_counts()
'''
small    257
large    105 '''
pd.crosstab(Ytest,pred_test)
'''
col_0          large  small
size_category              
large              6     99
small             10    247 '''

metrics.accuracy_score(Ytest,pred_test) # 0.6988
print(metrics.classification_report(Ytest,pred_test))
pd.set_option('display.max_columns' , 50)

'''
with hidden_layer_sizes=(30,30)
overall training accuracy is 85%, small 99% correctly classified and
large 35% correctly classified (recall)
overall testing accuracy is 70%, small 96% correctly classified and
large only 6% correctly classified (recall)

with hidden_layer_sizes=(50,50,50)
overall training accuracy is 97%, small 99% (120/121) correctly classified and
large 88% (30/34) correctly classified (recall)
overall testing accuracy is 65%, small 84% (215/257) correctly classified and
large only 21% (22/105) correctly classified (recall)
'''

################# using MLPRegressor
# here instead of size_category, we will use area which is continuous variable
fire = pd.read_csv('E:/Neural Networks/forestfires.csv')
fire.columns
fire.shape # 517 obs, 31 vars
fire.drop(['month','day','size_category'], axis=1, inplace=True)
fire.shape # (517, 28)

X1 = fire.drop(['area'], axis=1)
Y1 = fire['area']

Xtrain1, Xtest1,Ytrain1,Ytest1 = train_test_split(X1,Y1, test_size=0.7, random_state=123)

# standardizing predictors
scaler = StandardScaler()
scaler.fit(Xtrain1[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] )
Xtrain1[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] = scaler.transform(Xtrain1[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']])
Xtrain1.describe()
Xtest1[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']] = scaler.transform(Xtest1[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']])
Xtest1.describe()

### running the model
from sklearn.neural_network import MLPRegressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,),activation="relu", solver='adam',random_state=123)
mlp_regressor
mlp_regressor.fit(Xtrain1, Ytrain1)

## predicting
pred_train1 = mlp_regressor.predict(Xtrain1)
pred_test1 = mlp_regressor.predict(Xtest1)

metrics.mean_squared_error(Ytest1, pred_test1) # mse is 5494.745
np.sqrt(metrics.mean_squared_error(Ytest1, pred_test1)) # rmse is 74.1265
metrics.mean_absolute_error(Ytest1, pred_test1) # mae is 18.1824
np.corrcoef(Ytest1, pred_test1) # 0.0809 very low corr, poor model

# when use MLP regressor, the predictions are very poor.

############## CONCLUSIONS

'''
We have used neural network model to classify and predict the forest fires.
We have standardized the data to improve the performance of the model.
we have used MLPClassifier to classify forest fires into small or large size.
we have also used MLPRegressor to predict the area of forest fires. 
The results with former were good but latter model showed poor performance.

'''

