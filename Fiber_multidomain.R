setwd("/Users/yan/Desktop/同步文件夹/FIBER_update/FIBER_IRT")
library(pROC)
library(data.table)
library(jsonlite)
library(knitr)
source("Fiber_documented.R")
K=5
Y <- rep(c(0, 1), each = 4000, times = 4) ## True label: 0: human, 1: AI
score <- read.csv("./all_detector_scores_32000_14.csv")
fast_score <- score$fast_detect_gpt
Y_post <- matrix(0,ncol=K,nrow=25600)
fiber_result <- list()
Y_post_stream <- matrix(0,ncol=K,nrow=25600)
fiber_result_stream <- list()
tr <- tr_stream <- rep(0,K)
for(f in 1:K){
  D <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/D_32000_13.csv"))
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))

  s_train <- fast_score[-test_id]
  s_test <- fast_score[test_id]
  Y_train <- Y[-test_id]
  Y_test <- Y[test_id]
  D_train <- D[-test_id,]
  D_test <- D[test_id,]
  irt <- fromJSON(paste0("./cv_5fold_d5/fold_",f,"/irt_13_d5/best_parameters.json"))
  z_alpha <- irt$disc
  z_beta <- irt$diff
  z_train <- cbind(z_alpha,z_beta)
  theta<- irt$ability
  Y_pred<- fiber(s_train, 
                  Y_train, 
                  D_train, 
                  z_train, 
                  s_train,
                  D_train,
                  a_0=1,
                  b_0=1,
                  theta_det=theta
                )
  Y_post[,f] <- Y_pred$posterior
  Y_1 <- eval_with_best_threshold(Y_train,Y_post[,f])
  fiber_result[[f]]<- Y_1
  tr[f] <- as.numeric(unlist(Y_1$result[1]))[1]

  Y_pred_stream<- fiber_stream(Y_train, 
                                D_train, 
                                z_train, 
                                s_train, 
                                D_train,
                                a_0=1,
                                b_0=1,
                                theta_det=theta,
                              )
  Y_post_stream[,f] <- Y_pred_stream$posterior
  Y_stream <- eval_with_best_threshold(Y_train,Y_post_stream[,f])
  fiber_result_stream[[f]]<- Y_stream
  tr_stream[f] <- unlist(Y_stream$result[1])[1]
}

write.csv(Y_post,"fiber_posterior_train.csv")
write.csv(Y_post_stream,"fiber_stream_posterior_train.csv")

accuracy_fiber <- rep(0,K)
precision_fiber <- rep(0,K)
recall_fiber <- rep(0,K)
F1_fiber <- rep(0,K)
auc_fiber <- rep(0,K)
thre_fiber <- rep(0,K)
accuracy_fiber_stream <- rep(0,K)
precision_fiber_stream <- rep(0,K)
recall_fiber_stream <- rep(0,K)
F1_fiber_stream <- rep(0,K)
auc_fiber_stream <- rep(0,K)
thre_fiber_stream <- rep(0,K)

for(f in 1:K){
  accuracy_fiber[f]<- fiber_result[[f]]$result$Accuracy[1]
  precision_fiber[f]<- fiber_result[[f]]$result$Precision[1]
  recall_fiber[f]<- fiber_result[[f]]$result$Recall[1]
  F1_fiber[f]<- fiber_result[[f]]$result$F1[1]
  auc_fiber[f] <- fiber_result[[f]]$result$AUC[1]
  thre_fiber[f] <- fiber_result[[f]]$result$threshold[1]
  accuracy_fiber_stream[f]<- fiber_result_stream[[f]]$result$Accuracy[1]
  precision_fiber_stream[f]<- fiber_result_stream[[f]]$result$Precision[1]
  recall_fiber_stream[f]<- fiber_result_stream[[f]]$result$Recall[1]
  F1_fiber_stream[f]<- fiber_result_stream[[f]]$result$F1[1]
  auc_fiber_stream[f] <- fiber_result_stream[[f]]$result$AUC[1]
  thre_fiber_stream[f] <- fiber_result_stream[[f]]$result$threshold[1]
}
fiber_metric <- c(mean(accuracy_fiber),mean(precision_fiber),mean(recall_fiber),
    mean(F1_fiber),mean(auc_fiber))
fiber_metric_stream <- c(mean(accuracy_fiber_stream),
    mean(precision_fiber_stream),mean(recall_fiber_stream),mean(F1_fiber_stream),
  mean(auc_fiber_stream))
fiber_metric
thre_fiber
fiber_metric_stream
thre_fiber_stream

Y_post_test <- matrix(0,ncol=K,nrow=6400)
fiber_result_test <- list()
Y_post_stream_test <- matrix(0,ncol=K,nrow=6400)
Y_pred_result_test <- matrix(0,ncol=K,nrow=6400)
Y_pred_stream_result_test <- matrix(0,ncol=K,nrow=6400)

fiber_result_stream_test <- list()
tr_test <- tr_stream_test <- rep(0,K)
for(f in 1:K){
  D <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/D_32000_13.csv"))
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))

  s_train <- fast_score[-test_id]
  s_test <- fast_score[test_id]
  Y_train <- Y[-test_id]
  Y_test <- Y[test_id]
  D_train <- D[-test_id,]
  D_test <- D[test_id,]
  irt <- fromJSON(paste0("./cv_5fold_d5/fold_",f,"/irt_13_d5/best_parameters.json"))
  z_alpha <- irt$disc
  z_beta <- irt$diff
  z_train <- cbind(z_alpha,z_beta)
  theta<- irt$ability
  Y_pred<- fiber(s_train, 
                  Y_train, 
                  D_train, 
                  z_train, 
                  s_test,
                  D_test,
                  a_0=1,
                  b_0=1,
                  threshold = thre_fiber[f],
                  theta_det=theta     
                )
  Y_post_test[,f] <- Y_pred$posterior
  Y_pred_result_test[,f] <- Y_pred$y_pred
#  Y_1 <- eval_with_best_threshold(Y_train,Y_post[,f])
##fiber_result[[f]]<- Y_1
#  tr[f] <- as.numeric(unlist(Y_1$result[1]))[1]

  Y_pred_stream<- fiber_stream(Y_train, 
                                D_train, 
                                z_train, 
                                s_test, 
                                D_test,
                                a_0=1,
                                b_0=1,
                                threshold = thre_fiber_stream[f],
                                theta_det=theta
                              )
  Y_post_stream_test[,f] <- Y_pred_stream$posterior
  Y_pred_stream_result_test[,f] <- Y_pred_stream$y_pred
#  Y_stream <- eval_with_best_threshold(Y_train,Y_post_stream[,f])
#  fiber_result_stream[[f]]<- Y_stream
# tr_stream[f] <- unlist(Y_stream$result[1])[1]
}

write.csv(Y_post_test,"fiber_posterior_test.csv")
write.csv(Y_post_stream_test,"fiber_stream_posterior_test.csv")
TP <- FP<- FN <- TN <- rep(0,K)
TP_stream <- FP_stream<- FN_stream <- TN_stream <- rep(0,K)
for(f in 1:K){
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))
  Y_test <- Y[test_id]
  TP[f] <- sum(Y_pred_result_test[,f]==1&Y_test==1)
  FP[f] <- sum(Y_pred_result_test[,f]==1&Y_test==0)
  FN[f] <- sum(Y_pred_result_test[,f]==0&Y_test==1)
  TN[f] <- sum(Y_pred_result_test[,f]==0&Y_test==0)
  TP_stream[f] <- sum(Y_pred_stream_result_test[,f]==1&Y_test==1)
  FP_stream[f] <- sum(Y_pred_stream_result_test[,f]==1&Y_test==0)
  FN_stream[f] <- sum(Y_pred_stream_result_test[,f]==0&Y_test==1)
  TN_stream[f] <- sum(Y_pred_stream_result_test[,f]==0&Y_test==0)
}
ACC_fiber <- (TP+TN)/6400
PRE_fiber <- TP/(TP+FP)
REC_fiber <- TP/3200
F1 <- 2*PRE_fiber*REC_fiber/(PRE_fiber+REC_fiber)

ACC_fiber_stream <- (TP_stream+TN_stream)/6400
PRE_fiber_stream <- TP_stream/(TP_stream+FP_stream)
REC_fiber_stream <- TP_stream/3200
F1_stream <- 2*PRE_fiber_stream*REC_fiber_stream/(PRE_fiber_stream+REC_fiber_stream)


mean(ACC_fiber)
mean(PRE_fiber)
mean(REC_fiber)
mean(F1)
mean(ACC_fiber_stream)
mean(PRE_fiber_stream)
mean(REC_fiber_stream)
mean(F1_stream)

###########################

Y_post_short <- matrix(0,ncol=K,nrow=25600)
fiber_result_short <- list()
Y_post_stream_short <- matrix(0,ncol=K,nrow=25600)
fiber_result_stream_short <- list()
tr_short <- tr_stream_short <- rep(0,K)
for(f in 1:K){
  D <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/D_32000_13.csv"))
  D_short <- D[,-c(2,3,8)]
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))

  s_train <- fast_score[-test_id]
  s_test <- fast_score[test_id]
  Y_train <- Y[-test_id]
  Y_test <- Y[test_id]
  D_train_short <- D_short[-test_id,]
  D_test_short <- D_short[test_id,]
  irt_short <- fromJSON(paste0("./cv_5fold_d5/fold_",f,"/irt_10_d5/best_parameters.json"))
  z_alpha_short <- irt_short$disc
  z_beta_short <- irt_short$diff
  z_train_short <- cbind(z_alpha_short,z_beta_short)
  theta_short<- irt_short$ability
  Y_pred_short<- fiber(s_train, 
                  Y_train, 
                  D_train_short, 
                  z_train_short, 
                  s_train,
                  D_train_short,
                  a_0=1,
                  b_0=1,
                  theta_det=theta_short
                )
  Y_post_short[,f] <- Y_pred_short$posterior
  Y_1_short <- eval_with_best_threshold(Y_train,Y_post_short[,f])
  fiber_result_short[[f]]<- Y_1_short
  tr_short[f] <- as.numeric(unlist(Y_1_short$result[1]))[1]

  Y_pred_stream_short<- fiber_stream(Y_train, 
                                D_train_short, 
                                z_train_short, 
                                s_train, 
                                D_train_short,
                                a_0=1,
                                b_0=1,
                                theta_det=theta_short,
                              )
  Y_post_stream_short[,f] <- Y_pred_stream_short$posterior
  Y_stream_short <- eval_with_best_threshold(Y_train,Y_post_stream_short[,f])
  fiber_result_stream_short[[f]]<- Y_stream_short
  tr_stream_short[f] <- unlist(Y_stream_short$result[1])[1]
}

write.csv(Y_post_short,"fiber_posterior_short_train.csv")
write.csv(Y_post_stream_short,"fiber_stream_posterior_short_train.csv")

accuracy_fiber_short <- rep(0,K)
precision_fiber_short <- rep(0,K)
recall_fiber_short <- rep(0,K)
F1_fiber_short <- rep(0,K)
auc_fiber_short <- rep(0,K)
thre_fiber_short <- rep(0,K)
accuracy_fiber_stream_short <- rep(0,K)
precision_fiber_stream_short <- rep(0,K)
recall_fiber_stream_short <- rep(0,K)
F1_fiber_stream_short <- rep(0,K)
auc_fiber_stream_short <- rep(0,K)
thre_fiber_stream_short <- rep(0,K)

for(f in 1:K){
  accuracy_fiber_short[f]<- fiber_result_short[[f]]$result$Accuracy[1]
  precision_fiber_short[f]<- fiber_result_short[[f]]$result$Precision[1]
  recall_fiber_short[f]<- fiber_result_short[[f]]$result$Recall[1]
  F1_fiber_short[f]<- fiber_result_short[[f]]$result$F1[1]
  auc_fiber_short[f] <- fiber_result_short[[f]]$result$AUC[1]
  thre_fiber_short[f] <- fiber_result_short[[f]]$result$threshold[1]
  accuracy_fiber_stream_short[f]<- fiber_result_stream_short[[f]]$result$Accuracy[1]
  precision_fiber_stream_short[f]<- fiber_result_stream_short[[f]]$result$Precision[1]
  recall_fiber_stream_short[f]<- fiber_result_stream_short[[f]]$result$Recall[1]
  F1_fiber_stream_short[f]<- fiber_result_stream_short[[f]]$result$F1[1]
  auc_fiber_stream_short[f] <- fiber_result_stream_short[[f]]$result$AUC[1]
  thre_fiber_stream_short[f] <- fiber_result_stream_short[[f]]$result$threshold[1]
}
fiber_metric_short <- c(mean(accuracy_fiber_short),mean(precision_fiber_short),mean(recall_fiber_short),
    mean(F1_fiber_short),mean(auc_fiber_short))
fiber_metric_stream_short <- c(mean(accuracy_fiber_stream_short),
    mean(precision_fiber_stream_short),mean(recall_fiber_stream_short),mean(F1_fiber_stream_short),
  mean(auc_fiber_stream_short))
fiber_metric_short
thre_fiber_short
fiber_metric_stream_short
thre_fiber_stream_short

Y_post_test_short <- matrix(0,ncol=K,nrow=6400)
fiber_result_test_short <- list()
Y_post_stream_test_short <- matrix(0,ncol=K,nrow=6400)
Y_pred_result_test_short <- matrix(0,ncol=K,nrow=6400)
Y_pred_stream_result_test_short <- matrix(0,ncol=K,nrow=6400)

fiber_result_stream_test_short <- list()
tr_test_short <- tr_stream_test_short <- rep(0,K)
for(f in 1:K){
  D <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/D_32000_13.csv"))
  D_short <- D[,-c(2,3,8)]
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))

  s_train <- fast_score[-test_id]
  s_test <- fast_score[test_id]
  Y_train <- Y[-test_id]
  Y_test <- Y[test_id]
  D_train_short <- D_short[-test_id,]
  D_test_short <- D_short[test_id,]
  irt_short <- fromJSON(paste0("./cv_5fold_d5/fold_",f,"/irt_10_d5/best_parameters.json"))
  z_alpha_short <- irt_short$disc
  z_beta_short <- irt_short$diff
  z_train_short <- cbind(z_alpha_short,z_beta_short)
  theta_short<- irt_short$ability
  Y_pred_short<- fiber(s_train, 
                  Y_train, 
                  D_train_short, 
                  z_train_short, 
                  s_test,
                  D_test_short,
                  a_0=1,
                  b_0=1,
                  threshold = thre_fiber_short[f],
                  theta_det=theta_short,     
                )
  Y_post_test_short[,f] <- Y_pred_short$posterior
  Y_pred_result_test_short[,f] <- Y_pred_short$y_pred
#  Y_1 <- eval_with_best_threshold(Y_train,Y_post[,f])
##fiber_result[[f]]<- Y_1
#  tr[f] <- as.numeric(unlist(Y_1$result[1]))[1]

  Y_pred_stream_short<- fiber_stream(Y_train, 
                                D_train_short, 
                                z_train_short, 
                                s_test, 
                                D_test_short,
                                a_0=1,
                                b_0=1,
                                threshold = thre_fiber_stream_short[f],
                                theta_det=theta_short,
                              )
  Y_post_stream_test_short[,f] <- Y_pred_stream_short$posterior
  Y_pred_stream_result_test_short[,f] <- Y_pred_stream_short$y_pred
#  Y_stream <- eval_with_best_threshold(Y_train,Y_post_stream[,f])
#  fiber_result_stream[[f]]<- Y_stream
# tr_stream[f] <- unlist(Y_stream$result[1])[1]
}

write.csv(Y_post_test_short,"fiber_posterior_short_test.csv")
write.csv(Y_post_stream_test_short,"fiber_stream_posterior_short_test.csv")


TP_short <- FP_short<- FN_short <- TN_short <- rep(0,K)
TP_short_stream <- FP_short_stream<- FN_short_stream <- TN_short_stream <- rep(0,K)
for(f in 1:K){
  test_id <- read.csv(paste0("./cv_5fold_d5/fold_",f,"/test_sample_ids.csv"))
  test_id <- test_id+1
  test_id <- unlist(as.vector(test_id))
  Y_test <- Y[test_id]
  TP_short[f] <- sum(Y_pred_result_test_short[,f]==1&Y_test==1)
  FP_short[f] <- sum(Y_pred_result_test_short[,f]==1&Y_test==0)
  FN_short[f] <- sum(Y_pred_result_test_short[,f]==0&Y_test==1)
  TN_short[f] <- sum(Y_pred_result_test_short[,f]==0&Y_test==0)
  TP_short_stream[f] <- sum(Y_pred_stream_result_test_short[,f]==1&Y_test==1)
  FP_short_stream[f] <- sum(Y_pred_stream_result_test_short[,f]==1&Y_test==0)
  FN_short_stream[f] <- sum(Y_pred_stream_result_test_short[,f]==0&Y_test==1)
  TN_short_stream[f] <- sum(Y_pred_stream_result_test_short[,f]==0&Y_test==0)
}
ACC_fiber_short <- (TP_short+TN_short)/6400
PRE_fiber_short <- TP_short/(TP_short+FP_short)
REC_fiber_short <- TP_short/3200
F1_short <- 2*PRE_fiber_short*REC_fiber_short/(PRE_fiber_short+REC_fiber_short)

ACC_fiber_stream_short <- (TP_short_stream+TN_short_stream)/6400
PRE_fiber_stream_short <- TP_short_stream/(TP_short_stream+FP_short_stream)
REC_fiber_stream_short <- TP_short_stream/3200
F1_stream_short <- 2*PRE_fiber_stream_short*REC_fiber_stream_short/(PRE_fiber_stream_short+REC_fiber_stream_short)

mean(ACC_fiber_short)
mean(PRE_fiber_short)
mean(REC_fiber_short)
mean(F1_short)
mean(ACC_fiber_stream_short)
mean(PRE_fiber_stream_short)
mean(REC_fiber_stream_short)
mean(F1_stream_short)


############### domain_specific
