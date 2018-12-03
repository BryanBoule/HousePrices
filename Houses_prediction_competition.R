#-------------------------------------------------------------------
# -----------------------HOUSE PREDICTIONS--------------------------
# ------------------------------------------------------------------

  # Set path
  getwd()
  setwd("C:/Users/Bryan/Desktop/Housepred")

#-------------------------------------------------------------------
# --------------------------IMPORT PACKAGES-------------------------
# ------------------------------------------------------------------

  library(randomForest)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(corrplot)
  library(caret) # findCorrelation
  library(mice) # Imputation
  
#-------------------------------------------------------------------
# --------------------------IMPORT DATAS----------------------------
# ------------------------------------------------------------------

  # Import datas
  train <- read.csv(file= "train.csv", header=T, stringsAsFactors = F)
  test <- read.csv(file = "test.csv", header=T, stringsAsFactors = F)
  
  # Binary feature IsSale to caracterize whether an observation is part of the trainset
  train$isSale <- T
  test$isSale <- F
  
  # Add feature SalePrice on testset
  test$SalePrice <- NA
  
  # Observe the variable of interest
  qplot(SalePrice, data=train, bins=50)
  
  # As it is left squewed, applying a logtransformation the log-log form "draws in" big values which often makes the data 
  # easier to look at and normalizes the variance across observations.
  # Plus, If your data are log-normally distributed, then the log transformation makes them normally distributed. 
  # Normally distributed data have lots going for them.
  train$SalePrice <- log(train$SalePrice + 1)
  
  # Normale distribution now
  qplot(SalePrice, data=train, bins=50)
  
  # Merging datasets
  full <- rbind(train,test)
  
  colnames(full)
  
  # Id feature is useless
  full <- full[,-1]

#-------------------------------------------------------------------
# --------------------------VISUALISATION---------------------------
# ------------------------------------------------------------------

  glimpse(full)
  glimpse(train)
  
  ggplot(train, aes(x=MSSubClass, y=SalePrice))+
    geom_point()

  ggplot(train, aes(x=YrSold, y=SalePrice, group=YrSold))+
    geom_boxplot()
  # Count outliers
  sum((train$YrSold==2007))
  YrSold.2007 <- which(train$YrSold==2007)
  sum(train$SalePrice[YrSold.2007]>13.5)
  # In year 2007 : 2 outliers
  
  # Without boxplot
  ggplot(train, aes(x=YrSold, y=SalePrice))+
    geom_point()
  # A solution would be to add a threshold for observations above 13.5 and below 11 and suppress them

#-------------------------------------------------------------------
# ----------------------FEATURE SELECTION---------------------------
# ------------------------------------------------------------------

  # Display correlation matrix on numeric features only
  WhichNums <- sapply(train, is.numeric)
  Nums <- cor(train[WhichNums], use="pairwise.complete.obs") 
  #pairwise.complete.obs : Allow to compute only if datas in the two variables are not empty
  correlations <- corrplot(Nums, method="square", type = 'upper')
  
  # Find attributes that are highly corrected (ideally > 0.75)
  highlyCorrelated <- findCorrelation(Nums, cutoff=0.75)
  # print indexes of highly correlated attributes
  print(highlyCorrelated)
  
  # Display NAs
  missmap(train[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)
  
  # Number of NAs for each feature
  na_count.train <- sort(sapply(train[,1:ncol(train)], function(y) sum(is.na(y))), decreasing = T)

  na_count.test <- sort(sapply(test[,1:ncol(test)], function(y) sum(is.na(y))), decreasing = T)

  na_count <- na_count.test + na_count.train
  # Count number of features which have more than half NAs (-1 because IsSale is variable of interest)
  sum(na_count>dim(full)[1]/2) - 1 # dim(full)[1] gives number of total observations
  
  # Let's drop features on which there is more than half of NAs
  sum(is.na(full$FireplaceQu))
  # FireplaceQu could be dropped too
  RetireFeatures <- c("PoolQC", "MiscFeature", "Alley", "Fence")
  DiffFeatures <- setdiff(names(full), RetireFeatures)
  full.4exclude <- full[DiffFeatures]
  
  # Exclude correlated features
  names(full[WhichNums][highlyCorrelated])
  RetireFeaturesCor <- c("GrLivArea"  ,"X1stFlrSF",  "GarageCars", "YearBuilt" )
  DiffFeaturesCor <- setdiff(names(full.4exclude),RetireFeaturesCor)
  full.CorExclude <- full.4exclude[DiffFeaturesCor]
  
  # Convert char features to factor
  full.CorExclude[sapply(full.CorExclude, is.character)] <- lapply(full.CorExclude[sapply(full.CorExclude, is.character)], 
                                         as.factor)
  
  glimpse(full.CorExclude)
  
#-------------------------------------------------------------------
# ---------------------------NAs INPUT------------------------------
# ------------------------------------------------------------------
  
  # Imput NAs ATTENTION CHECK IF 4exclude or CorExclude
  # In the testset, SalePrice is imputed but it will be overwritten 
  imputation <- mice(full.CorExclude, m=1, meth='cart', printFlag=FALSE, seed=7)
  full.imputed <- complete(imputation)
  
  # Resplit
  test.imputed <- full.imputed[full.imputed$isSale == F,]
  train.imputed <- full.imputed[full.imputed$isSale == T,]
  
  train.cleared <- train.imputed[-which(names(train.imputed)=='isSale')]
  
  # Check again number of NAs
  na_count <- sort(sapply(full.imputed[,1:ncol(full.imputed)], function(y) sum(is.na(y))), decreasing = T)
  sum(na_count)
  
#-------------------------------------------------------------------
# ---------------------------MODELING-------------------------------
# ------------------------------------------------------------------
  
  # Model
  fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    verboseIter = T
    #repeat = 5
    )
  
  SalePrice.equation.all <- "SalePrice ~ ."
  SalePrice.formula <- as.formula(SalePrice.equation.all)
  
  set.seed(7)
  fitglmnet <- train(SalePrice.formula, data = train.cleared, 
                     method = "glmnet",
                     trControl = fitControl,
                     na.action=na.pass)
  
  set.seed(7)
  fitglmboost <- train(SalePrice.formula, data = train.cleared, 
                 method = "glmboost", 
                 trControl = fitControl,
                 tuneGrid = expand.grid(mstop = c(50, 100, 150, 200, 250, 300),
                             prune = c('yes', 'no')),
                 na.action=na.pass)
  
  set.seed(7)
  # Find optimal mtry
  tuneRF(x = train.cleared[,-which(colnames(train.cleared)=="SalePrice")], y = train.cleared$SalePrice, nTreeTry = 250, stepFactor = 1.5, improve = 0.01) 
  fitrf <- train(SalePrice.formula, 
                 data = train.cleared, 
                 method = "cforest", 
                 trControl = fitControl,
                 na.action=na.pass,
                 tuneGrid = expand.grid(mtry=c(23)))
  
  # Display results
  results <- resamples(list(glmnet=fitglmnet, glmboost = fitglmboost, rf = fitrf), metric="RMSE")
  dotplot(results, metric="RMSE")
  summary(results)
  # GLMNET is the best algorithm

#-------------------------------------------------------------------
# -------------------------PREDICTION-------------------------------
# ------------------------------------------------------------------

  # Apply inverse transformation of the previous logtransformation
  test.imputed$SalePrice <- exp(predict(fitglmnet, test.imputed))-1
  warnings()
  SalePrice <- test.imputed$SalePrice
  
  #-------------------------------------------------------------------
  # ---------------------------RESULT---------------------------------
  # ------------------------------------------------------------------
  
  # Create the result dataframe
  output <- as.data.frame(test$Id)
  output$SalePrice <- SalePrice
  colnames(output) <- c("Id", "SalePrice")
  
  # Export the result dataframe
  write.csv(output, file = "kaggle_submission.csv", row.names = F)

#-------------------------------------------------------------------
# -----------------------------END----------------------------------
# ------------------------------------------------------------------

# Thank you for reading, do not hesitate to comment or contact me so that i can improve my results
# bryanboule@gmail.com