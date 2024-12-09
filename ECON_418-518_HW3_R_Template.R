##################################################
# ECON 418-518 Homework 3
# Kate Jaramillo
# The University of Arizona
# kmjaramillo@arizona.edu 
# 8 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table, caret, randomForest)
library(data.table, caret, randomForest)

# Set seed
set.seed(418518)

# Loading data in
csv_data <- read.csv("Econ_418-518_HW3_Data.csv")

#####################
# Problem 1
#####################


#################
# Part (i)
#################

# Dropping columns
data <- csv_data[, !names(csv_data) %in% c("fnlwgt", "occupation", "relationship", 
                                       "capital-gain", "capital-loss", "educational-num")]

##############
# Part (ii)
##############

# Converting income to binary variable
data$income <- ifelse(data$income == ">50K", 1, 0)

# Converting race to binary variable
data$race <- ifelse(data$race == "White", 1, 0)

# Gender to binary variable
data$gender <- ifelse(data$gender == "Male", 1, 0)

# Workclass to binary variable
data$workclass <- ifelse(data$workclass == "Private", 1, 0)

# Native.country to binary variable
data$native.country <- ifelse(data$native.country == "United-States", 1, 0)

# Marital.status to binary variable
data$marital.status <- ifelse(data$marital.status == "Married-civ-spouse", 1, 0)

# Education to binary variable
data$education <- ifelse(data$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

# New age_sq variable
data$age_sq <- data$age^2

# Standardize age
data$age_std <- scale(data$age)

# Standardize age_sq
data$age_sqstd <- scale(data$age_sq)

# Standardize hours.per.week
data$hours.per.week_std <- scale(data$hours.per.week)

#################
# Part (iii)
#################

# Proportion of obs w/ income > 50K
income_prop <- mean(data$income == 1)

# Proportion of obs in private sector
privsect_prop <- mean(data$workclass == 1)

# Proportion of obs married
married_prop <- mean(data$marital.status == 1)

# Proportion of obs female
fem_prop <- mean(data$gender == 0)

# Number of NAs 
total_na <- sum(is.na(data))

# Convert income to factor data
data$income <- factor(data$income)

#################
# Part (iv)
#################

# Find last training set obs
train_size <- floor(0.7 * nrow(data))

# Create training data set
train_set <- data[1:train_size, ]

# Create testing data set
test_set <- data[(train_size + 1):nrow(data), ]

##################
# Part (v)
##################

# Install and load glmnet for ridge/lasso
install.packages("glmnet")
library(glmnet)

# Create feature matrix
X_train <- model.matrix(income ~ ., train_set)[, -1]

# Create outcome vector
y_train <- train_set$income

# Lambda grid
grid <- 10^seq(5, -2, length = 50)

# Lasso model with train function
lasso_model <- train(
  x = X_train, y = y_train,
  method = "glmnet",
  data = train_set, 
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = grid)
)

lasso_model

# Coefficient estimates
lasso_coefs <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
lasso_coefs

# Subset data w/ non-zero vars
reduced_vars <- c("age", "education", "marital.status", "race",
                  "capital.gain", "capital.loss", "hours.per.week", "income")
new_data <- train_set[, reduced_vars]

# Create new lambda grid
grid <- 10^seq(5, -2, length = 50)

# Train lasso reg
lasso_model2 <- train(
  income ~ .,
  data = new_data,
  method = "glmnet",
  trComtrol = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = grid)
)

# Train ridge reg
ridge_model <- train(
  income ~ .,
  data = new_data,
  method = "glmnet",
  trComtrol = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = grid)
)

# Best performance for lasso model
lasso_best <- lasso_model2$bestTune
lasso_best_acc <- lasso_model2$results$Accuracy[lasso_model2$results$lambda == lasso_best$lambda]
cat("Lasso Best Acc: ", lasso_best_acc, "\n")

# Best performance for ridge reg
ridge_best <- ridge_model$bestTune
ridge_best_acc <- ridge_model$results$Accuracy[ridge_model$results$lambda == ridge_best$lambda]
cat("Ridge Best Acc: ", ridge_best_acc, "\n")

#################
# Part (vi)
#################

# Define grid of mtry values
mtry_grid <- expand.grid(mtry = c(2, 5, 9))

# Create list to store models
models_rf <- list()

# Create three random forests with diff num of trees
for (t in c(100, 200, 300))
{
  # Number of trees in forest
  print(paste0(t, "trees in the forest."))
  
  # Define the model
  model_rf <- train(
    income ~ .,
    data = train_set,
    method = "rf",
    tuneGrid = mtry_grid,
    trControl = trainControl(method = "cv", number = 5),
    ntree = t,
    importance = TRUE
  )
  
  # Store model in list
  models_rf[[paste0("ntree_", t)]] <- model_rf

  # Show model
  print(model_rf)

  print("-----------------------------")  
}

# Pull out best model
best_rf <- model_rf$bestTune
print(best_rf)

# Predict on training set 
train_predict <- predict(best_rf, newdata = train_set)





