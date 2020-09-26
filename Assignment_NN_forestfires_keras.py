##### Assignment Neural Networks on forestfires dataset - KERAS SEQUENTIAL

# Business Problem: To classify the burned area of forest fires, small or large 

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

fire = pd.read_csv('E:/Neural Networks/forestfires.csv')
fire.columns
fire.dtypes

fire.drop(['month','day','area'],  axis=1, inplace=True)
fire.shape # 517 and 28 var

fire.isnull().sum() # no missing values

# size_category is the dependant variable
fire.size_category.value_counts() # small  378 , large  139

#  small as 0 and large as 1
fire.loc[fire.size_category=='small', 'size_category'] =0
fire.loc[fire.size_category=='large', 'size_category'] =1
fire.size_category.value_counts() # small  378 , large  139

train,test = train_test_split(fire,test_size = 0.3,random_state=123)

Xtrain = train.drop(["size_category"],axis=1)
Ytrain = train["size_category"]
Xtest = test.drop(["size_category"],axis=1)
Ytest = test["size_category"]

# Defining the structure for ANN network 
def prep_model(hidden_dim):
    model = Sequential() 
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    # To define the dimensions for the output layer
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    # to define loss function , optimizer, metrics
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model    

# giving input as list format which is referring to
# [input features=27, hidden_layer1=50, HL2=40, HL3=20, output_layer=1]

Xtrain.shape # 27 variables
keras_model1 = prep_model([27,100,100,100,1])

# Fitting ANN model with epochs = 100 
keras_model1.fit(np.array(Xtrain),np.array(Ytrain),epochs=1000,batch_size=50)

# with [27,50,40,20,1]
# epochs=10,  loss: 0.6586 - accuracy: 0.6731
# epochs=100, loss: 0.5190 - accuracy: 0.7590
# epochs=500 loss: 0.4256 - accuracy: 0.7978

# with [27,100,100,100,1]
# epochs=1000, batch-size=50, loss: 0.3305 - accuracy: 0.8476

# with [27,40,40,40,1]
# epochs =500, loss: 0.3836 - accuracy: 0.8144
# epochs=1000, batch-size=50, loss: 0.2687 - accuracy: 0.8670

## plotting the model
from keras.utils import plot_model
import pydot
import pydotplus
import graphviz
from pydotplus import graphviz

plot_model(keras_model1,to_file="keras_model1.png")

# Predicting for test data 
pred_test = keras_model1.predict(np.array(Xtest))

pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["small"]*156)
pred_test_class[[i>=0.5 for i in pred_test]] = "large"

test["original_class"] = "small"
test.loc[test.size_category==1,"original_class"] = "large"
test.original_class.value_counts()  # small 113    large 43
test.size_category.value_counts()

# evaluating test data
pd.crosstab(pd.Series(test.original_class).reset_index(drop=True),pred_test_class)
metrics.accuracy_score(pd.Series(test.original_class).reset_index(drop=True),pred_test_class) # 0.628
# np.mean(pd.Series(test.original_class).reset_index(drop=True) == pred_test_class)
print(metrics.classification_report(pd.Series(test.original_class).reset_index(drop=True),pred_test_class))

# with [27,40,40,40,1], epochs=1000, batch-size=50,
# overall accuracy is 63%
# correct classification of small is 78% and that of large is just 23%

#### CONCLUSIONS
'''
We have used forestfires dataset and employed neural network model to
predict the size(small, large) or the area of forest fires.

first we have used neural network packages from sklearn: 
MLPClassifier was used to classify forest fires into small or large sizes.
MLPRegresor was used to predict the area of forest fires. This model gave 
poor results.
Then we used keras package - Sequential model. 
The results of MLPClassifier and keras-sequential model were similar. 
we used various combinations of hidden layers and neurons. Though the
overall testing accuracy was between 63% to 70%, correctly classifying the
large forest fires was a challenge.

'''


