library(glmnet)
library(readxl)
library(randomForest)
library(ggplot2)
library(tidyverse)
library(reshape)
library(caret)
library(pROC)
library(grid)
library(gridExtra)

# Returns a named list of models from a list of four fit models: 
# randomForest, ridge, lasso, and elnet

list_of_models <- function(models_list){
  
  names_list <- list()
  
  for (model in models_list){
    if (class(model)[1] == "randomForest")  {name  <- "randomForest"} 
    else if (model$call$alpha == 0)         {name   <-  'ridge'} 
    else if (model$call$alpha == 0.5)       {name   <-  'elnet'} 
    else if (model$call$alpha == 1)         {name   <-  'lasso'}
    
    names_list <- append(names_list, name)
  }
  
  names(models_list) <- names_list
  return(models_list)
}

# Returns a single row data.frame containing evaluation metrics for a fitted model
# in a particular iteration (i) 
# Takes a fitted model, X.data, Y.data, the name of the fitted model, a threshold
# and the i value as arguments
## Create vectors to store CV compute time
rid.cv.t = vector(); las.cv.t = vector(); elnet.cv.t = vector(); 

calculate_metrics <- function(fit, X.data, Y.data, fit_name, thrs, i){
  
  n  <- nrow(X.data)
  X  <- as.matrix(X.data)
  #Y.data <- factor(Y.data, levels = c("0", "1"), ordered = is.ordered(Y.data))
  
  
  if (class(fit)[1] == "randomForest"){
    y.hat  <- predict(fit, X)
  } else {
    # same as predict function
    beta0.hat  <- fit$a0
    beta.hat  <- as.vector(fit$beta)
    prob  <- exp(X %*% beta.hat +  beta0.hat  )/(1 + exp(X %*% beta.hat +  beta0.hat  )) 
    y.hat  <- ifelse(prob > thrs, 1, 0)
  }
  
  
  FP  <- sum(Y.data[y.hat==1] == 0) # false positives = negatives in the data that were predicted as positive
  TP  <- sum(y.hat[Y.data==1] == 1) # true positives = positives in the data that were predicted as positive
  P  <- sum(Y.data==1) # total positives in the data
  N  <- sum(Y.data==0) # total negatives in the data
  FPR  <- FP/N # false positive rate = type 1 error = 1 - specificity
  TPR  <- TP/P # true positive rate = 1 - type 2 error = sensitivity    
  AUC  <- auc(Y.data, as.numeric(y.hat), levels = c(0, 1), direction = "<")
  
  errors = data.frame(i, fit_name, FPR, TPR, AUC)
  return(errors)
}

data <- read.csv("https://raw.githubusercontent.com/jacob-bayer/STA9891-Final-Project/develop/features.csv")
#definitions <- read.csv("https://raw.githubusercontent.com/jacob-bayer/STA9891-Final-Project/develop/features_definitions.csv", row.names = 1)

drops <- c("NNP","WRB")
data <- data[ , !(names(data) %in% drops)]

Y <- (data["Label"] == 'objective')*1
X <- data[,4:60] 

## standardize data (not used currently)

X <- X %>% mutate_all(~(scale(.) %>% as.vector))

n  <- nrow(X)
p  <- ncol(X)

train  <- sample(n, n * 0.9)
X.train  <- X[train,]
X.test  <- X[-train,]

Y.train  <- Y[train,]
Y.test  <- Y[-train,]

n.train  <- nrow(X.train)
n.test  <- nrow(X.test)

indices  <- list()
metrics.train.all <- data.frame(matrix(ncol = 5, nrow = 0))
metrics.test.all <- data.frame(matrix(ncol = 5, nrow = 0))

for (i in 1:50){
  
  print(i)
  
  learn.index  <- sample(n.train, n.train * 0.9)
  
  #store indices used in each fold for future reference
  indices[i]   <-  list(learn.index)
  
  X.learn  <- as.matrix(X.train[learn.index,])
  X.validation  <- as.matrix(X.train[-learn.index,])
  Y.learn  <- Y.train[learn.index]
  Y.validation  <- Y.train[-learn.index]
  
  rf_start <- Sys.time()
  rf.fit  <- randomForest(y = as.factor(Y.learn), x = (X.learn), mtry = sqrt(p), nodesize=10)
  rf_end <- Sys.time()
  
  lasso_start <- Sys.time()
  lasso.cv  <- cv.glmnet(X.learn, Y.learn, family = "binomial", alpha = 1,  intercept = TRUE,  nfolds = 10, type.measure="class")
  lasso.cv.start <- Sys.time()
  lasso  <- glmnet(X.learn, Y.learn, lambda = lasso.cv$lambda.min, family = "binomial", alpha = 1,  intercept = TRUE)
  lasso.cv.end <- Sys.time()
  lasso_end <- Sys.time()
  
  elnet_start <- Sys.time()
  elnet.cv  <- cv.glmnet(X.learn, Y.learn, family = "binomial", alpha = 0.5,  intercept = TRUE,  nfolds = 10, type.measure="class")
  elnet.cv.start <- Sys.time()
  elnet  <- glmnet(X.learn, Y.learn, lambda = elnet.cv$lambda[which.min(elnet.cv$cvm)], family = "binomial", alpha = 0.5,  intercept = TRUE)
  elnet.cv.end <- Sys.time()
  elnet_end <- Sys.time()
  
  ridge_start <- Sys.time()
  ridge.cv  <- cv.glmnet(X.learn, Y.learn, family = "binomial", alpha = 0,  intercept = TRUE,   nfolds = 10, type.measure="class")
  ridge.cv.start <- Sys.time()
  ridge  <- glmnet(X.learn, Y.learn, lambda = ridge.cv$lambda[which.min(ridge.cv$cvm)], family = "binomial", alpha = 0,  intercept = TRUE)
  ridge.cv.end <- Sys.time()
  ridge_end <- Sys.time()
  
  fit_times <- data.frame("randomForest" = rf_end    - rf_start,
                          "lasso"        = lasso_end - lasso_start,
                          "elnet"        = elnet_end - elnet_start,
                          "ridge"        = ridge_end - ridge_start)
  las.cv.runtime = lasso.cv.end - lasso.cv.start
  rid.cv.runtime = ridge.cv.end - ridge.cv.start
  elnet.cv.runtime = elnet.cv.end - elnet.cv.start
  
  # Calculate Metrics
  models <- list_of_models(list(rf.fit, lasso, elnet, ridge))
  
  rid.cv.t = rbind(rid.cv.t, rid.cv.runtime)
  las.cv.t = rbind(las.cv.t, las.cv.runtime)
  elnet.cv.t = rbind(elnet.cv.t, elnet.cv.runtime)
  
  for (fit_name in names(models)){
    
    fit           <- models[fit_name][[1]]
    
    training_metrics <- calculate_metrics(fit, X.learn, Y.learn, fit_name, thrs = 0.5, i)
    training_metrics$training_time <- fit_times[fit_name][[1]]
    #training_metrics$cv_time <- cv_times[fit_name][[1]]
    
    val_metrics <- calculate_metrics(fit, X.validation, Y.validation, fit_name, thrs = 0.5, i)
    val_metrics$training_time <- fit_times[fit_name][[1]]
    
    
    metrics.train.all <- rbind(metrics.train.all, training_metrics)
    
    metrics.test.all  <- rbind(metrics.test.all, val_metrics)
    
    
  }
  
}


metrics.train.all <- metrics.train.all %>% 
  group_by(fit_name) %>% 
  arrange(fit_name)

metrics.test.all <- metrics.test.all %>% 
  group_by(fit_name) %>% 
  arrange(fit_name)

make_auc_boxplots <- function(metrics.data.frame, title){
  ggplot(metrics.data.frame, aes(fit_name, AUC, color = fit_name)) +
    geom_boxplot()+
    ylim(0.7,1)+
    ggtitle(title) +
    #xlab("Model") +
    theme(plot.title = element_text(hjust=0.5))+
    theme(axis.text.x = element_text(color = "grey20", size = 10, hjust = .5, vjust = .5, face = "plain"),
          axis.title.y = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain")) + 
    theme(#axis.title.y = element_blank(), 
      axis.title.x = element_blank())+
    theme(legend.position="none")
}

train.boxplot <- make_auc_boxplots(metrics.train.all, "Training AUCs by Model")
test.boxplot  <- make_auc_boxplots(metrics.test.all,  "Test AUCs by Model")

all_boxplots <- grid.arrange(train.boxplot,test.boxplot, nrow = 1)
plot(all_boxplots)

# Plot CV Curves

#CV curves
par(mfcol = c(1, 3))

# LASSO
plot(lasso.cv)
title(line = 2, sub = "Lasso", adj=0)

# RIDGE
plot(ridge.cv)
title(line = 2, sub = "Ridge", adj=0)

# ELASTIC NET
plot(elnet.cv)
title(line = 2, sub = "Elnet", adj=0)


cv.t.data <- cbind(rid.cv.t,las.cv.t,elnet.cv.t) 

colnames(cv.t.data) = c("Ridge.CV.Time",
                        "Lasso.CV.Time",
                        "Elnet.CV.Time"
)

avg.cv.t <- colMeans(cv.t.data)
ridge.cv.time <- avg.cv.t[1] #Ridge
lasso.cv.time <- avg.cv.t[2] #Lasso
elnet.cv.time <- avg.cv.t[3] #Elnet
cv.times <- rbind(ridge.cv.time,lasso.cv.time,elnet.cv.time)
colnames(cv.times) = "CV Runtime"
row.names(cv.times) = c("Ridge","Lasso","Elnet")
cv.times

# Time it takes to cross validate model

mean_times <- metrics.train.all %>% summarize("Mean Training Time" = mean(training_time, na.rm=TRUE))
median_test_aucs <- metrics.test.all %>% summarize("Median Test AUCs" = median(AUC, na.rm=TRUE))

median_auc_and_mean_time <- merge(median_test_aucs, mean_times)
median_auc_and_mean_time

importances <- data.frame(name = colnames(X.train))

for (fit_name in names(models)){
  fit           <- models[fit_name][[1]]
  
  if (fit_name == "randomForest"){
    importances[fit_name]  <- fit$importance[,1]
  } else {
    importances[fit_name]  <- fit$beta[,1]
  }
  
}

importances <- importances %>% 
  arrange(desc(elnet)) %>%
  mutate(order = seq(nrow(importances))) %>%
  melt(id = c("name", "order"))

elnet_importances <- importances %>% filter(variable == "elnet")
ridge_importances <- importances %>% filter(variable == "ridge")
lasso_importances <- importances %>% filter(variable == "lasso")
rf_importances    <- importances %>% filter(variable == "randomForest")

#Use same order for Lasso, Ridge, RF, create 4x1 figure
elnet_importances$feature     =  factor(elnet_importances$name, levels = elnet_importances$name[order(elnet_importances$value, decreasing = TRUE)])
lasso_importances$feature     =  factor(lasso_importances$name, levels = elnet_importances$name[order(elnet_importances$value, decreasing = TRUE)])
ridge_importances$feature     =  factor(ridge_importances$name, levels = elnet_importances$name[order(elnet_importances$value, decreasing = TRUE)])
rf_importances$feature        =  factor(rf_importances$name, levels = elnet_importances$name[order(elnet_importances$value, decreasing = TRUE)])

col_elnet = ifelse(elnet_importances$value>0,"turquoise2","tomato1")
col_lasso = ifelse(lasso_importances$value>0,"turquoise2","tomato1")
col_ridge = ifelse(ridge_importances$value>0,"turquoise2","tomato1")

elnetPlot =  ggplot(elnet_importances, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill=col_elnet, colour="black",) +
  ggtitle("Elnet Variable Importances") + labs(x="",y="Coefficients")   +
  theme(axis.text.x = element_blank())

lassoPlot =  ggplot(lasso_importances, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill=col_lasso, colour="black") +
  ggtitle("Lasso Variable Importances") + labs(x="",y="Coefficients")  +
  theme(axis.text.x = element_blank())

ridgePlot =  ggplot(ridge_importances, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill=col_ridge, colour="black") +
  ggtitle("Ridge Variable Importances") + labs(x="",y="Coefficients")  +
  theme(axis.text.x = element_blank())

rfPlot =  ggplot(rf_importances, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="turquoise2", colour="black") +
  ggtitle("Random Forest Variable Importances") +labs(x="",y="Coefficients") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

## For the slide we prepared we prepared the above with x-axis element blank and exported the importance order to a csv,
## rotated and placed below the graph for readability.  For the above you will need to zoom or export for readability unless your plot area is large.

grid.arrange(elnetPlot, lassoPlot, ridgePlot, rfPlot, nrow = 4)

## write.csv(elnet_importances, file="F:\\Masters Classes\\STA 9891 - Machine Learning for Data Mining\\Final Project\\elnet features.csv")
