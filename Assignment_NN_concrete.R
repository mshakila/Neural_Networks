############### ASSIGNMENT NEURAL NETWORKS for concrete data

### Business problem: To build a Neural Network model to predict concrete strength 

concrete <- read.csv("E:\\Neural Networks\\concrete.csv")
head(concrete)
names(concrete)
str(concrete) # all are numeric variables
summary(concrete)

normalize <- function(x){
  return((x - min(x)) / (max(x)-min(x)))
}
a=c(1,2,3,4,5)
normalize(a)

concrete_norm <- as.data.frame(lapply(concrete, normalize))
summary(concrete_norm)

# splitting the data
library(caTools)
set.seed(123)
concrete_split <- sample.split(concrete_norm$strength, SplitRatio=0.7)
concrete_train <- subset(concrete_norm, concrete_split==TRUE) # 721 obs of 9 vars
concrete_test <- subset(concrete_norm, concrete_split==FALSE) # 309 obs of 9 vars

library(neuralnet)
concrete_model1 <- neuralnet(strength ~ . , data=concrete_train)
plot(concrete_model1)
model_results1 <- compute(concrete_model1, concrete_test[1:8])
predicted_strength1 <- model_results1$net.result
# evaluating model performance
cor(predicted_strength1, concrete_test$strength) # 0.8173
mean((predicted_strength1 - concrete_test$strength)^2) # mse is 0.01396

concrete_model2 <- neuralnet(strength ~ . , data=concrete_train, hidden=c(5,2))
plot(concrete_model2)
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result

cor(predicted_strength2, concrete_test$strength) # 0.93525
mean((predicted_strength2 - concrete_test$strength)^2) # mse is 0.005297

# just 1 perceptron, cor is 0.8173 and mse is 0.01396
# hidden c(5,2) cor is 0.93365 and mse is 0.0053414
# hidden c(5,5,4) cor is 0.93525 and mse is 0.005297

############### ANN using functional API
####### functional API has 2 parts: inputs and outputs

library(keras)
class(concrete_train)
concrete_train_data <- data.matrix(concrete_train[1:8]) # converting to matrix
concrete_train_labels <- as.array(concrete_train[,9]) # converting to array
class(concrete_train_data)
class(concrete_train_labels)


####### input layer

dim(concrete_train_data) # 721 obs and 8 variables
inputs <- layer_input(shape = dim(concrete_train_data)[2])

# outputs compose of inputs + dense layers
predictions <- inputs %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 1) # o/p layer no activation since it is regr prob

## create and compile model
model1 <- keras_model(inputs = inputs, outputs = predictions)
model1 %>% compile(
  optimizer = "rmsprop" ,
  loss = "mse" ,
  metrics = list("mean_absolute_error")
)

model1 %>% fit(concrete_train_data, concrete_train_labels,epochs=500,batch_size=100)
# after 30th epoch of batch-size 100, loss(mse): 0.0104 - mean_absolute_error: 0.0804
# after 50th epoch of batch-size 100, loss(mse): 0.0055 - mean_absolute_error: 0.0578
# after 100th epoch of batch-size 100, loss(mse): 0.0043 - mean_absolute_error: 0.0518
# after 200th epoch of batch-size 100, loss(mse): 0.0018 - mean_absolute_error: 0.0312
# after 500th epoch of batch-size 100, loss(mse): 0.0013 - mean_absolute_error: 0.0270

## test performance
score1 <- model1 %>% evaluate(data.matrix(concrete_test[1:8]), as.array(concrete_test[,9]))
# after 30th epoch of batch-size 100, loss (mse):0.0115 - mean_absolute_error: 0.0834
# after 50th epoch of batch-size 100, loss (mse):0.0069 - mean_absolute_error: 0.0620
# after 100th epoch of batch-size 100, loss (mse):0.0049 - mean_absolute_error: 0.0506
# after 200th epoch of batch-size 100, loss (mse):0.0047 - mean_absolute_error: 0.0459
# after 500th epoch of batch-size 100, loss (mse):0.0056 - mean_absolute_error: 0.0481

# From 200th epoch to 500th epoch, 
# though training loss (mse) has decreased from 0.0018 to 0.0013,
# testing loss (mse) has increased from 0.0047 to 0.0056

#### CONCLUSIONS
'''
We have to predict the strength of concrete based on various predictors. 
We have used neural net package and keras package to predict.

For this dataset, both models have given similar results.

'''
