### Capstone House Price Prediction
### author: XuMan


# 1 Preparation

options(scipen=100)

library(tidyverse)
library(caret)
library(Rborist)
library(FNN)
library(glmnet)
library(gbm)
library(moments)
library(corrplot)

train_set <- read.csv("/Users/xuman/ds_projects/House-Price-Prediction/house-prices-advanced-regression-techniques/train.csv")
test_set <- read.csv("/Users/xuman/ds_projects/House-Price-Prediction/house-prices-advanced-regression-techniques/test.csv")

test_set$SalePrice <- NA
dataset <- rbind(train_set, test_set)

# 2 Dataset exploration

## 2.1 Data structure
dim(train_set)

## 2.2 Clean the dataset

### 2.2.1 Find the variables with NA
perct_na <- sort(desc(apply(dataset, 2, function(x) sum(is.na(x))/nrow(dataset)*100)))
perct_na[perct_na < 0]

### 2.2.2 Features of missing data

# We can see that NAs in "GarageType","GarageQual","GarageCond","GarageFinish", "GarageYrBlt", "BsmtExposure","BsmtFinType2","BsmtQual","BsmtCond","BsmtFinType1", "MasVnrType" because there is no garage, basement or Masonry veneer
dataset %>% select(starts_with("Garage")) %>%
  arrange(GarageArea) %>%
  top_n(100, desc(GarageArea)) %>%
  head()

### 2.2.3 Deal with NAs
# 1) ignore variables with more than 20% of missing values: "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"
# 2) fill the NAs in "GarageType","GarageQual","GarageCond","GarageFinish", "GarageYrBlt", "BsmtExposure","BsmtFinType2","BsmtQual","BsmtCond","BsmtFinType1", "MasVnrType" with "None"
# 3) fill NAs in "GarageYrBlt" with the year the house was built "YearBuilt"
# 4) fill NAs in "LotFrontage" with the median of available values;
# 5) fill NAs in "MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea" with 0
# 5) fill NAs in "Electrical", "MSZoning" with the most frequent type 
# 6) for the variable "Utilities", since 2916 houses are "AllPub" and only 1 house is "NoSeWa", if we impute ‘AllPub’ for the NAs, the variable will be useless  for prediction. So I simply remove it

var_names_na_to_none <- c("GarageType","GarageQual","GarageCond","GarageFinish", "BsmtExposure","BsmtFinType2","BsmtQual","BsmtCond","BsmtFinType1", "MasVnrType")
var_names_na_to_zero <- c("MasVnrArea", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea")
fill_with_highest_req <- c("MSZoning", "Functional", "Exterior1st", "Exterior2nd", "KitchenQual", "Electrical", "SaleType")

dataset_2 <- dataset %>%
  select(-c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "Utilities")) %>%
  mutate_at(var_names_na_to_none, function(x) ifelse(is.na(x), "None", x)) %>%
  mutate(GarageYrBlt = ifelse(is.na(GarageYrBlt), YearBuilt, GarageYrBlt)) %>%
  mutate(LotFrontage = ifelse(is.na(LotFrontage), median(LotFrontage, na.rm = TRUE), LotFrontage)) %>%
  mutate_at(var_names_na_to_zero, function(x) ifelse(is.na(x), 0, x)) %>%
  mutate_at(fill_with_highest_req, function(x) ifelse(is.na(x), levels(x)[which.max(table(x))], x))

# verify that there is no NA left
sapply(dataset_2, function(x) sum(is.na(x)))

### 2.2.4 Standarize variable type

sapply(dataset_2, class)

# Some variables should be defined as factor

var_names_to_type_factor <- c("MSSubClass", "MSZoning", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "Exterior1st", "Exterior2nd", "MoSold", "YrSold", "MasVnrType", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", "Electrical", "KitchenQual", "Functional", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "SaleType")

dataset_3 <- dataset_2 %>% mutate_at(var_names_to_type_factor, function(x) as.factor(x))

sapply(dataset_3, class)

## 2.3 Explore the variables

### 2.3.1 Distribution of Sale Price

dataset_3 %>% ggplot(aes(SalePrice)) +
  geom_histogram(color = "white")

skewness(dataset_3$SalePrice, na.rm = TRUE)
skewness(log(dataset_3$SalePrice), na.rm = TRUE)

## 2.4 Normalize the dataset
numeric_predictors <- dataset_3[index_numeric]
numeric_predictors <- numeric_predictors[, -which(colnames(numeric_predictors) %in% c("Id","SalePrice"))]
predictors_skewness <- sapply(numeric_predictors, skewness)
vars_to_normalize <- names(predictors_skewness[abs(predictors_skewness) > 0.8])

dataset_4 <- dataset_3 %>%
  mutate_at(vars_to_normalize, function(x) log(x+1))

train_set_4 <- dataset_4[!is.na(dataset_4$SalePrice), ]
test_set_4 <- dataset_4[is.na(dataset_4$SalePrice), ]

### 3 Analysis

## 3.1 Random Forest

control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
set.seed(40)
train_rf <- train(log(SalePrice)~ .-Id, data = train_set_4,  method = "Rborist",  trControl = control,  tuneGrid = data.frame(predFixed = 2, minNode = c(3, 5, 10, 25, 50)))
plot(train_rf)

y_train <- log(train_set_4$SalePrice)
x_train <- train_set_4 %>% select(-c("Id", "SalePrice"))
fit_rf <- Rborist(x_train, y_train, minNode = 3)

x_test <- test_set_4 %>% select(-c("Id", "SalePrice"))
y_predict_rf <- predict(fit_rf, x_test)
y_hat_rf <- exp(y_predict_rf[["yPred"]])

sub_rf <- data.frame(Id = test_set$Id, SalePrice = y_hat_rf)
write.csv(sub_rf, file = "sub_rf_2.csv", row.names = FALSE)

## 3.2 Lasso
x_train_lasso <- model.matrix(as.formula( log(SalePrice)~ .-Id ), train_set_4)

set.seed(40)
fit_lasso <- cv.glmnet(x_train_lasso, y_train, alpha=1)
test_set_4$SalePrice <- 1
x_test_lasso <- model.matrix(as.formula( log(SalePrice)~ .-Id ), test_set_4)
y_predict_lasso <- predict(fit_lasso, newx = x_test_lasso, s = "lambda.min")
y_hat_lasso <- exp(as.numeric(y_predict_lasso))

sub_lasso <- data.frame(Id = test_set$Id, SalePrice = y_hat_lasso)
write.csv(sub_lasso, file = "sub_lasso.csv", row.names = FALSE)

## 3.3 GBDT
train_gbdt <- train(log(SalePrice)~ .-Id, data = train_set_4,  method = "gbm",  trControl = control)
y_predict_gbdt <- predict(train_gbdt, test_set_4)
y_hat_gbdt <- exp(y_predict_gbdt)
sub_gbdt <- data.frame(Id = test_set$Id, SalePrice = y_hat_gbdt)
write.csv(sub_gbdt, file = "sub_gbdt.csv", row.names = FALSE)

## 3.4 average the random forest, lasso and gbdt model

y_predict_rf_lasso_gbdt <- (y_hat_rf + y_hat_lasso + y_hat_gbdt)/3
sub_rf_lasso_gbdt <- data.frame(Id = test_set$Id, SalePrice = y_predict_rf_lasso_gbdt)
write.csv(sub_rf_lasso_gbdt, file = "sub_rf_lasso_gbdt.csv", row.names = FALSE)