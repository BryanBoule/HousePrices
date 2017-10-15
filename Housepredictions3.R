setwd("C:/Users/Bryan/Desktop/Housepred")

#-------------------------------------------------------------------
# --------------------------IMPORT PACKAGES-------------------------
# ------------------------------------------------------------------

install.packages("dplyr")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("caret")
install.packages("rpart")
install.packages("randomForest")
install.packages("gridExtra")
install.packages("DMwR")
install.packages("mice")
install.packages("Amelia")
install.packages("robustbase")
install.packages("mboost")
install.packages("plyr")
install.packages("import")
install.packages("mlbench")

library(mlbench)
library(mboost)
library(plyr)
library(import)
library(robustbase)
library(Amelia)
library(mice)
library(DMwR)
library(gridExtra)
library(rpart)
library(randomForest)
library(caret)
library(corrplot)
library(dplyr)
library(ggplot2)

#-------------------------------------------------------------------
# -------------------IMPORT DONNEES ET FUSION-----------------------
# ------------------------------------------------------------------

train <- read.csv(file= "train.csv", header=T, stringsAsFactors = F)
test <- read.csv(file = "test.csv", header=T, stringsAsFactors = F)

train$isSale <- T
test$isSale <- F

test$SalePrice <- NA

qplot(SalePrice, data=train, bins=50)
train$SalePrice <- log(train$SalePrice + 1)
qplot(SalePrice, data=train, bins=50)

full <- rbind(train,test)
full <- full[,-1]

#-------------------------------------------------------------------
# --------------------------VISUALISATION---------------------------
# ------------------------------------------------------------------

glimpse(full)
glimpse(train)

p1 <- ggplot(train, aes(x=LotFrontage, y=SalePrice)) +
  geom_point()+
  geom_smooth(method = lm) # 2 outliers (935,1299 in train)

p2 <- ggplot(train, aes(x=LotArea, y=SalePrice))+
  geom_point()+
  geom_smooth(method = lm)

grid.arrange(p1, p2, ncol=2, nrow = 1)
#____________________________________

ggplot(train, aes(x=MSSubClass, y=SalePrice))+
  geom_point()
#discretize MSSubClass

ggplot(train, aes(x=YrSold, y=SalePrice, group=YrSold))+
  geom_boxplot()
#2 outliers
#2007

ggplot(train, aes(x=YrSold, y=SalePrice))+
  geom_point()

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

#Plot la matrice de corrélation sur les features type numeric
WhichNums <- sapply(train, is.numeric)
Nums <- cor(train[WhichNums], use="pairwise.complete.obs") 
#pairwise.complete.obs : cet argument permet de faire le calcul seulement si il y a des données dans les deux variables
correlations <- corrplot(Nums, method="square", type = 'upper')

#feature importance METHODE1
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(Nums, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#Affichage des NAs
missmap(train[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)

#nb de NA values dans chaque colonne
na_count <- sort(sapply(train[,1:ncol(train)], function(y) sum(is.na(y))), decreasing = T)
na_count

#on exclue les features comportant plus de la moitié de NA
RetireFeatures <- c("PoolQC", "MiscFeature", "Alley", "Fence")
DiffFeatures <- setdiff(names(full), RetireFeatures)
full.4exclude <- full[DiffFeatures]

#on exclut les features corréles
names(full[WhichNums][highlyCorrelated])
RetireFeaturesCor <- c("GrLivArea"  ,"X1stFlrSF",  "GarageCars", "YearBuilt" )
DiffFeaturesCor <- setdiff(names(full.4exclude),RetireFeaturesCor)
full.CorExclude <- full[DiffFeaturesCor]

# converti char features en factor
full.CorExclude[sapply(full.CorExclude, is.character)] <- lapply(full.CorExclude[sapply(full.CorExclude, is.character)], 
                                       as.factor)
#skip feat correlees suppression
full.4exclude[sapply(full.4exclude, is.character)] <- lapply(full.4exclude[sapply(full.4exclude, is.character)], 
                                                                 as.factor)


glimpse(full.CorExclude)

#Imput NAs ATTENTION CHECK IF 4exclude or CorExclude
imputation <- mice(full.4exclude, m=1, meth='cart', printFlag=FALSE, seed=7)
full.imputed <- complete(imputation)

#Resplit
test.imputed <- full.imputed[full.imputed$isSale == F,]
train.imputed <- full.imputed[full.imputed$isSale == T,]

train.cleared <- train.imputed[-which(names(train.imputed)=='isSale')]

#nb de NA values dans chaque colonne
na_count <- sort(sapply(train.cleared[,1:ncol(train.cleared)], function(y) sum(is.na(y))), decreasing = T)
na_count

#modelisation + appr
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  verboseIter = T,
  ## repeated ten times
  repeats = 5)

SalePrice.equation.all <- "SalePrice ~ ."
SalePrice.formula <- as.formula(SalePrice.equation.all)

set.seed(7)
fitglmnet <- train(SalePrice.formula, data = train.cleared, 
                   method = "glmnet",
                   trControl = fitControl,
                   na.action=na.pass)

set.seed(7)
fitlasso <- train(SalePrice.formula, data = train.cleared, 
                   method = "lasso", 
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
fitrf <- train(SalePrice.formula, 
               data = train.cleared, 
               method = "cforest", 
               trControl = fitControl,
               na.action=na.pass,
               tuneGrid = expand.grid(mtry=c(200,250)))

#afficher resultats
results <- resamples(list(glmnet=fitglmnet, lasso=fitlasso), metric="RMSE")
dotplot(results, metric="RMSE")
summary(results)

#Choix du meilleur algo
test.imputed$SalePrice <- exp(predict(fitglmnet, test.imputed))-1
SalePrice <- test.imputed$SalePrice

#Mettre au format attendu
output <- as.data.frame(test$Id)
output$SalePrice <- SalePrice
colnames(output) <- c("Id", "SalePrice")

#Exporter les données
write.csv(output, file = "kaggle_submission.csv", row.names = F)
