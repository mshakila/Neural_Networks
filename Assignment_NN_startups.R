############### ASSIGNMENT NEURAL NETWORKS

### Business problem: To build a Neural Network model for 50_startups data to predict profit 

library(neuralnet)
library(nnet)

startups <- read.csv("E:\\Neural Networks\\50_Startups.csv")
head(startups)
table(startups$State)

# since 'state' has 3 categories convert to dummy variables
library(psych)
State_dummy <- dummy.code(startups$State, group = NULL)
startups <- as.data.frame(cbind(startups, State_dummy))
head(startups,3)
 names(startups)[1] <- "RD_Spend"
 names(startups)[3] <- "Marketing_Spend"
 names(startups)[7] <- "NewYork"
names(startups)

# finding missing values in dataset
startups[!complete.cases(startups),] # no missing values

# since neural nets can work with large datasets, and have many hidden layers
# and many neurons, its always better to normalize the data.
# normalize 
normalize <- function(x){
  return((x-min(x)) / (max(x) - min(x)))
}

startups_norm <- as.data.frame(lapply(startups[,c(1,2,3,5)], normalize))
summary(startups_norm) # all variable values range btw 0 and 1

startups_norm <- as.data.frame(cbind(startups_norm,startups[,c(6,7,8)]))
names(startups_norm)
head(startups_norm)

#### creating training and testing data
library(caTools)
set.seed(123)
sample <- sample.split(startups_norm$Profit, SplitRatio = 0.8)
train <- subset(startups_norm, sample==TRUE)
test <- subset(startups_norm, sample==FALSE)
# since only 50 datapoints, we are using more (80%) data to train the model

### train the model
# startups_model1 <- neuralnet(Profit ~ RD_Spend+Administration+Marketing_Spend 
#                             + california+NewYork+Florida, data=train)
names(train)
startups_model1 <- neuralnet(Profit ~ ., data=train)
# error 0.069044 and steps 110
startups_model1$model.list$variables

# plot the model
plot(startups_model1)

#evaluate results
model_results1 <- compute(startups_model1, test[-4])
pred_profit1 <- model_results1$net.result

# accuracy can also be found using correlation
cor(pred_profit1, test$Profit) # 0.9529
plot(pred_profit1, test$Profit, col='blue')

mean((pred_profit1 - test$Profit)^2) 
# MSE is 0.002578016, very low, indicating that it is a good model

#### Improving model performance
startups_model2 <- neuralnet(Profit~., data=train, hidden=c(3,3,2))
plot(startups_model2)
model_results2 <- compute(startups_model2, test[-4])
pred_profit2 <- model_results2$net.result
cor(pred_profit2, test$Profit)
plot(pred_profit2, test$Profit, col='red')
mean((pred_profit2 - test$Profit)^2) # 

################ Linear Regression Model
startups_linreg <- lm(Profit~.-NewYork, data=train)
summary(startups_linreg)
# Multiple R-squared:  0.9499,	Adjusted R-squared:  0.9425 
pred_linreg <- predict(startups_linreg, newdata=test[-4])
cor(pred_linreg, test$Profit) # 0.9611


# CONCLUSIONS
'''
We have used the data of 50 start-up companies to predict the Profit.
We have used Neural networks model. By using just 1 perceptron, we got good
accuracy of 95%. Since the data is very less, by increasing the hidden layers
or by increasing the number of neurons per layer, has not added much to the
accuracy of the model.

When we compared NN model with multi-linear regression model, latter has better
accuracy.

To find the real power of NN, we have to have large datasets and many variables.
Running NN on Small data results in overfitting the model.

'''

