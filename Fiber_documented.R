# =============================================================================
# FIBER: IRT-informed Bayesian Evidence Recalibration for LLM Text Detection
# =============================================================================
# Research implementation of FIBER and FIBER-Stream, together with utilities
# for selecting classification thresholds and visualizing cross-validated ROC
# curves. The implementation is intended for reproducible research workflows.
#
# Label convention:
#   0 = human-written text (H); 1 = AI-generated text (A).
#
# Required packages:
#   mvtnorm  - multivariate Gaussian densities for historical IRT summaries
#   pROC     - ROC curves, AUC, and operating-threshold selection
#
# Before calling fiber() or fiber_stream(), fit the IRT model separately to the
# labeled historical detector-correctness matrix. Provide its fitted item
# parameters as z_train / Z_train, with rows aligned to the historical labels.
# Detector-output matrices contain binary predictions (0/1), with matching
# detector columns and column order in the historical and incoming datasets.
# Fast-DetectGPT scores are continuous and must be aligned to their samples.
# Its binary detector decision should NOT also appear among the auxiliary
# detector outputs when reproducing the paper's FIBER configuration.
#
# This file documents the original research code; it does not change the
# statistical procedures, convergence rules, or existing return structures.
# =============================================================================

library(mvtnorm)

#' Evaluate the class-conditional likelihood of auxiliary detector outputs.
#'
#' Assumes conditional independence of binary detector decisions given class.
#' The likelihood is a product of Bernoulli probabilities, evaluated on the
#' log scale before exponentiation to reduce intermediate numerical underflow.
#'
#' @param p Numeric vector of positive-prediction probabilities, one per detector.
#'   Values should be strictly between 0 and 1 (e.g., after additive smoothing).
#' @param d Binary matrix with one text per row and one detector per column.
#'   The column order must match the entries in `p`.
#' @return Numeric likelihood vector, one value per text.

detector_likelihood <- function(p, d) {
  p <- as.numeric(p)
  
  d <- as.matrix(d)
  
  # Multiply independent Bernoulli terms in log space; exponentiate at return.
  log_h <- d %*% log(p) + (1 - d) %*% log(1 - p)
  
  as.numeric(exp(log_h))
}

#' Apply FIBER using labeled historical Fast-DetectGPT scores.
#'
#' Fits class-conditional Gaussian models for historical IRT item parameters
#' and Fast-DetectGPT scores; constructs an IRT-informed empirical-Bayes-style
#' prior; and combines that prior with the score and auxiliary detector evidence
#' to obtain posterior probabilities for new texts.
#'
#' @param s_train Continuous Fast-DetectGPT scores for historical texts.
#' @param y_train Historical binary labels (0 = human, 1 = AI-generated).
#' @param d_train Historical binary detector-output matrix (rows = texts).
#' @param z_train Historical IRT item-parameter matrix, row-aligned to `y_train`.
#' @param s_test Continuous Fast-DetectGPT scores for new texts.
#' @param d_test Detector-output matrix for new texts. Columns must match
#'   `d_train` in both identity and order.
#' @param w Numeric vector of three evidence exponents: historical prior,
#'   Fast-DetectGPT score density, and auxiliary detector likelihood.
#'   The default `c(1, 1, 1)` gives the unweighted Bayesian decomposition.
#' @param a_0,b_0 Positive Beta-prior hyperparameters for the AI-class share.
#' @param threshold Classification cutoff applied to the posterior probability.
#' @return List with `posterior`, `y_pred`, `pi_irt`, historical soft
#'   associations `eta`, weights `w`, estimated `parameters`, and class-wise
#'   `likelihoods` for the new texts.
#' @note This function uses fitted IRT parameters for historical texts only;
#'   test-text IRT parameters are neither required nor estimated here.

fiber <- function(
  s_train,
  y_train,
  d_train,
  z_train,
  s_test,
  d_test,
  w=c(1,1,1),
  a_0 = 1,
  b_0 = 1,
  threshold = 0.5
) {
  
  s_train <- as.numeric(s_train)
  y_train <- as.numeric(y_train)
  d_train <- as.matrix(d_train)
  z_train <- as.matrix(z_train)
  s_test  <- as.numeric(s_test)
  d_test  <- as.matrix(d_test)
  
  n_sup <- length(y_train)
  n_un  <- length(s_test)
  
  # Partition the historical sample by known class labels.
  idx_A <- which(y_train == 1)
  idx_H <- which(y_train == 0)
  
  n_A <- length(idx_A)
  n_H <- length(idx_H)
  
  if (n_A < 2 || n_H < 2) {
    stop("Both AI and human classes must have at least two samples.")
  }
  
  # Gaussian approximation to IRT item parameters within each class.
  mu_A_z <- colMeans(z_train[idx_A, , drop = FALSE])
  mu_H_z <- colMeans(z_train[idx_H, , drop = FALSE])
  
  Sigma_A_z <- cov(z_train[idx_A, , drop = FALSE])
  Sigma_H_z <- cov(z_train[idx_H, , drop = FALSE])
  
  # Evaluate each historical IRT representation under both class models.
  dens_A_z <- dmvnorm(
    z_train,
    mean = mu_A_z,
    sigma = Sigma_A_z
  )
  
  dens_H_z <- dmvnorm(
    z_train,
    mean = mu_H_z,
    sigma = Sigma_H_z
  )
  
  # Form class-association soft counts using the Beta-prior mean.
  eta_A <- a_0/(a_0+b_0) * dens_A_z
  eta_H <- b_0/(a_0+b_0) * dens_H_z
  
  eta <- eta_A / (eta_A + eta_H)
  
  # Aggregate historical IRT associations into the empirical-Bayes-style prior.
  pi_irt <- (a_0 + sum(eta)) / (a_0 + b_0 + n_sup)
  
  # Fit Gaussian score densities using labeled historical scores.
  mu_A_s <- mean(s_train[idx_A])
  mu_H_s <- mean(s_train[idx_H])
  
  sigma2_A_s <- var(s_train[idx_A])
  sigma2_H_s <- var(s_train[idx_H])
  
  # Laplace smoothing (Beta(1, 1)) prevents exact 0/1 Bernoulli estimates.
  alpha_smooth <- 1

  p_A <- (colSums(d_train[idx_A, , drop = FALSE]) + alpha_smooth) /
  (n_A + 2 * alpha_smooth)

  p_H <- (colSums(d_train[idx_H, , drop = FALSE]) + alpha_smooth) /
  (n_H + 2 * alpha_smooth)
  # Evaluate the score and detector evidence for the new texts.
  g_A_s <- dnorm(
    s_test,
    mean = mu_A_s,
    sd = sqrt(sigma2_A_s)
  )
  
  g_H_s <- dnorm(
    s_test,
    mean = mu_H_s,
    sd = sqrt(sigma2_H_s)
  )
  
  h_A_d <- detector_likelihood(p_A, d_test)
  h_H_d <- detector_likelihood(p_H, d_test)
  
  # Evidence exponents allow optional weighting of the three log-odds terms.
  numerator <- pi_irt^w[1] * g_A_s^w[2] * h_A_d^w[3]
  
  denominator <- numerator +
    (1 - pi_irt)^w[1] * g_H_s^w[2] * h_H_d^w[3]
  
  posterior <- numerator / denominator
  
  # Convert AI posterior probabilities to hard labels using the supplied cutoff.
  y_pred <- as.numeric(posterior > threshold)
  
  return(list(
    posterior = posterior,
    y_pred = y_pred,
    pi_irt = pi_irt,
    eta = eta,
    w=w,
    parameters = list(
      Z = list(
        mu_A_z = mu_A_z,
        mu_H_z = mu_H_z,
        Sigma_A_z = Sigma_A_z,
        Sigma_H_z = Sigma_H_z
      ),
      s = list(
        mu_A_s = mu_A_s,
        mu_H_s = mu_H_s,
        sigma2_A_s = sigma2_A_s,
        sigma2_H_s = sigma2_H_s
      ),
      detector = list(
        p_A = p_A,
        p_H = p_H
      )
    ),
    likelihoods = list(
      g_A_s = g_A_s,
      g_H_s = g_H_s,
      h_A_d = h_A_d,
      h_H_d = h_H_d
    )
  ))
}

#' Estimate class-conditional Fast-DetectGPT score distributions by EM.
#'
#' Fits a two-component Gaussian mixture for incoming unlabeled scores while
#' treating the historical IRT prior and detector-output likelihoods as fixed.
#' Initialization uses soft AI-class membership probabilities provided in `w0`.
#'
#' @param w0 Initial AI-class responsibilities, one per incoming text.
#' @param s Continuous Fast-DetectGPT scores for the incoming batch.
#' @param h_ad,h_hd Detector-output likelihoods under the AI and human classes.
#' @param pi_irt Fixed IRT-informed AI-class prior probability.
#' @param max_iter Maximum number of EM iterations (default: 500).
#' @param tol Absolute observed-data log-likelihood change used for stopping.
#' @param verbose Whether to print progress every 10 EM iterations.
#' @return List of estimated class means and variances, final responsibilities
#'   `w`, the EM trace, terminal log-likelihood, iteration count, and convergence
#'   flag. Despite their names, `sigma_as` and `sigma_hs` are VARIANCES, not SDs.
#' @note The algorithm uses the orientation that larger scores tend to indicate
#'   AI-generated text. This function works on a batch, not one observation at a time.

fiber_stream_em <- function(
  w0,
  s,
  h_ad,
  h_hd,
  pi_irt,
  max_iter = 500,
  tol = 1e-6,
  verbose = FALSE
) {
  
  w <- as.numeric(w0)
  s <- as.numeric(s)
  h_ad <- as.numeric(h_ad)
  h_hd <- as.numeric(h_hd)
  
  n <- length(s)
  # Small numerical offset used to avoid zero denominators in responsibilities.
  eps <- 1e-12
  sum_wA <- sum(w)
  sum_wH <- sum(1 - w)
  
  if (sum_wA < eps || sum_wH < eps) {
    stop("Initial weights are degenerate. Please check w0.")
  }
  
  # Soft memberships initialize both class-specific Gaussian distributions.
  mu_as <- sum(w * s) / sum_wA
  mu_hs <- sum((1 - w) * s) / sum_wH
  
  sigma_as <- sum(w * (s - mu_as)^2) / sum_wA
  sigma_hs <- sum((1 - w) * (s - mu_hs)^2) / sum_wH

  
  # Keep the AI-associated Gaussian at the higher score mean.
  if (mu_as < mu_hs) {
    tmp <- mu_as
    mu_as <- mu_hs
    mu_hs <- tmp
    
    tmp <- sigma_as
    sigma_as <- sigma_hs
    sigma_hs <- tmp
    
    w <- 1 - w
  }
  
  iter <- 0
  diff <- Inf
  loglik_old <- -Inf
  
  # Record the observed-data log-likelihood and estimated parameters.
  trace <- data.frame(
    iter = integer(),
    loglik = numeric(),
    diff = numeric(),
    mu_as = numeric(),
    sigma_as = numeric(),
    mu_hs = numeric(),
    sigma_hs = numeric()
  )
  
  while (diff > tol && iter < max_iter) {
    
    iter <- iter + 1
    
    g_as <- dnorm(s, mean = mu_as, sd = sqrt(sigma_as))
    g_hs <- dnorm(s, mean = mu_hs, sd = sqrt(sigma_hs))
    
    # E-step: update AI-class responsibilities from current Gaussian densities.
    comp_A <- pi_irt * g_as * h_ad
    comp_H <- (1 - pi_irt) * g_hs * h_hd
    
    w <- comp_A / (comp_A + comp_H + eps)
    
    sum_wA <- sum(w)
    sum_wH <- sum(1 - w)
    
    if (sum_wA < eps || sum_wH < eps) {
      warning("One component has nearly zero weight. EM stopped.")
      break
    }
    
    # M-step: update weighted class means and variances.
    mu_as_new <- sum(w * s) / sum_wA
    mu_hs_new <- sum((1 - w) * s) / sum_wH
    
    sigma_as_new <- sum(w * (s - mu_as_new)^2) / sum_wA
    sigma_hs_new <- sum((1 - w) * (s - mu_hs_new)^2) / sum_wH 
    
    if (mu_as_new < mu_hs_new) {
      tmp <- mu_as_new
      mu_as_new <- mu_hs_new
      mu_hs_new <- tmp
      
      tmp <- sigma_as_new
      sigma_as_new <- sigma_hs_new
      sigma_hs_new <- tmp
      
      w <- 1 - w
    }
    
    mu_as <- mu_as_new
    mu_hs <- mu_hs_new
    sigma_as <- sigma_as_new
    sigma_hs <- sigma_hs_new
    
    g_as <- dnorm(s, mean = mu_as, sd = sqrt(sigma_as))
    g_hs <- dnorm(s, mean = mu_hs, sd = sqrt(sigma_hs))
    
    # Re-evaluate the observed-data mixture log-likelihood.
    loglik <- sum(log(
      pi_irt * g_as * h_ad +
        (1 - pi_irt) * g_hs * h_hd
    ))
    
    # Stop when ABSOLUTE (not relative) log-likelihood change is small.
    diff <- abs(loglik - loglik_old)
    
    trace <- rbind(
      trace,
      data.frame(
        iter = iter,
        loglik = loglik,
        diff = diff,
        mu_as = mu_as,
        sigma_as = sigma_as,
        mu_hs = mu_hs,
        sigma_hs = sigma_hs
      )
    )
    
    if (verbose && iter %% 10 == 0) {
      cat(
        "Iter:", iter,
        "loglik:", round(loglik, 6),
        "diff:", signif(diff, 4),
        "mu_as:", round(mu_as, 4),
        "mu_hs:", round(mu_hs, 4),
        "\n"
      )
    }
    
    loglik_old <- loglik
  }
  
  # Keep variances in sigma_* fields for compatibility with the original code.
  list(
    mu_as = mu_as,
    sigma_as = sigma_as,
    mu_hs = mu_hs,
    sigma_hs = sigma_hs,
    w = w,
    trace = trace,
    loglik = tail(trace$loglik, 1),
    iterations = iter,
    converged = diff <= tol
  )
}


#' Apply FIBER-Stream without historical Fast-DetectGPT scores.
#'
#' Reuses labeled historical detector outputs and fitted IRT item parameters to
#' estimate the historical prior and detector likelihoods. It then estimates
#' class-conditional Gaussian score distributions from an incoming unlabeled
#' batch using `fiber_stream_em()`.
#'
#' @param y_train Historical binary labels (0 = human, 1 = AI-generated).
#' @param D_train Historical auxiliary detector-output matrix (binary).
#' @param Z_train Historical fitted IRT item-parameter matrix, row-aligned to
#'   `y_train` and `D_train`.
#' @param s_stream Continuous Fast-DetectGPT scores for the incoming batch.
#' @param D_stream Binary detector outputs for the incoming batch; columns must
#'   match those in `D_train`.
#' @param pi_0 Initial class probability used when computing IRT associations.
#' @param a_0,b_0 Beta-prior hyperparameters used for the IRT-informed prior.
#' @param threshold Posterior classification cutoff.
#' @param max_iter Maximum EM iterations for the incoming batch.
#' @param verbose Whether to print EM iteration summaries.
#' @return List with `posterior`, `y_pred`, `pi_irt`, `eta`, detector-only initial
#'   responsibilities `omega_init`, EM fit `em`, fitted `parameters`, and detector
#'   `likelihoods`. The original code's `parameters$score` field accesses
#'   `em_fit$params`, which is not returned by `fiber_stream_em()`.
#' @note Historical Fast-DetectGPT scores are not required. Incoming texts must
#'   be processed together for the batchwise EM update.

fiber_stream <- function(
  y_train,
  D_train,
  Z_train,
  s_stream,
  D_stream,
  pi_0 = 0.5,
  a_0 = 1,
  b_0 = 1,
  threshold = 0.5,
  max_iter = 500,
  verbose = FALSE
) {
  
  y_train <- as.numeric(y_train)
  D_train <- as.matrix(D_train)
  Z_train <- as.matrix(Z_train)
  s_stream <- as.numeric(s_stream)
  D_stream <- as.matrix(D_stream)
  
  n_sup <- length(y_train)
  n_un <- length(s_stream)
  eps <- 1e-12 
  
  # Only historical labels are used to estimate supervised summaries.
  idx_A <- which(y_train == 1)
  idx_H <- which(y_train == 0)
  
  n_A <- length(idx_A)
  n_H <- length(idx_H)
  
  if (n_A < 2 || n_H < 2) {
    stop("Both classes must have at least two supervised samples.")
  }
  
  # Fit class-conditional Gaussian models to retained IRT item parameters.
  mu_A_Z <- colMeans(Z_train[idx_A, , drop = FALSE])
  mu_H_Z <- colMeans(Z_train[idx_H, , drop = FALSE])
  
  Sigma_A_Z <- cov(Z_train[idx_A, , drop = FALSE])
  Sigma_H_Z <- cov(Z_train[idx_H, , drop = FALSE])
  
  dens_A_Z <- dmvnorm(
    Z_train,
    mean = mu_A_Z,
    sigma = Sigma_A_Z
  )
  
  dens_H_Z <- dmvnorm(
    Z_train,
    mean = mu_H_Z,
    sigma = Sigma_H_Z
  )
  
  # Use pi_0 as the initial class probability for IRT soft association.
  eta_A <- pi_0 * dens_A_Z
  eta_H <- (1 - pi_0) * dens_H_Z
  eta <- eta_A / (eta_A + eta_H)
  
  # Convert soft historical associations into a fixed IRT-informed prior.
  pi_irt <- (a_0 + sum(eta)) / (a_0 + b_0 + n_sup)
  
  # Estimate Bernoulli detector rates with Laplace smoothing.
  alpha_smooth <- 1

  p_A <- (colSums(D_train[idx_A, , drop = FALSE]) + alpha_smooth) /
  (n_A + 2 * alpha_smooth)

  p_H <- (colSums(D_train[idx_H, , drop = FALSE]) + alpha_smooth) /
  (n_H + 2 * alpha_smooth)
  
  # Evaluate fixed historical detector likelihoods on incoming outputs.
  h_A <- detector_likelihood(p_A, D_stream)
  h_H <- detector_likelihood(p_H, D_stream)
  
  # Initialize EM responsibilities without historical score observations.
  omega_init <- (pi_irt * h_A) /
    (pi_irt * h_A + (1 - pi_irt) * h_H)
  
  # Adapt only the score-distribution parameters on the unlabeled batch.
  em_fit <- fiber_stream_em(
    w0 = omega_init,
    s = s_stream,
    h_ad = h_A,
    h_hd = h_H,
    pi_irt = pi_irt,
    max_iter = max_iter,
    tol = 1e-6,
    verbose = verbose
  )
  mu_as <- em_fit$mu_as
  mu_hs <- em_fit$mu_hs
  sigma_as <- em_fit$sigma_as
  sigma_hs <- em_fit$sigma_hs

  g_as <- dnorm(s_stream, mean = mu_as, sd = sqrt(sigma_as))
  g_hs <- dnorm(s_stream, mean = mu_hs, sd = sqrt(sigma_hs))
  
  # Final posterior combines adapted score densities and retained evidence.
  p_hat <- pi_irt * g_as * h_A /(pi_irt * g_as * h_A + (1 - pi_irt) * g_hs * h_H)
  y_pred <- as.numeric(p_hat > threshold)
  
  return(list(
    posterior = p_hat,
    y_pred = y_pred,
    pi_irt = pi_irt,
    eta = eta,
    omega_init = omega_init,
    em = em_fit,
    parameters = list(
      IRT = list(
        mu_A_Z = mu_A_Z,
        mu_H_Z = mu_H_Z,
        Sigma_A_Z = Sigma_A_Z,
        Sigma_H_Z = Sigma_H_Z
      ),
      detector = list(
        p_A = p_A,
        p_H = p_H
      ),
      # NOTE: `fiber_stream_em()` does not return a `params` element currently.
      score = em_fit$params
    ),
    likelihoods = list(
      h_A = h_A,
      h_H = h_H
    )
  ))
}

#' Select an ROC-based threshold and report binary-classification metrics.
#'
#' Uses pROC to find an operating threshold (Youden by default) and evaluates
#' accuracy, precision, recall, specificity, F1, and confusion-matrix counts at
#' that threshold. The score should increase with evidence for the AI class.
#'
#' @param y_true Binary reference labels (0 = human, 1 = AI-generated).
#' @param p_hat Numeric predicted scores or AI-class posterior probabilities.
#' @param method Threshold-selection criterion passed to `pROC::coords()`.
#' @return List with a one-row metric table `result`, the ROC object `roc`, and
#'   the binary prediction vector `y_pred`.
#' @note This routine SELECTS and EVALUATES a cutoff on the SAME labels.
#'   For unbiased held-out evaluation, select the cutoff using training-fold
#'   predictions and labels, then apply it to the test scores without using
#'   test labels to select the cutoff. Multiple tied optimal thresholds may
#'   require choosing one explicitly.

eval_with_best_threshold <- function(y_true, p_hat, method = "youden") {
  
  library(pROC)
  
  y_true <- as.numeric(y_true)
  p_hat <- as.numeric(p_hat)
  
  # ROC and threshold use the labels supplied to THIS function.
  roc_obj <- roc(response = y_true, predictor = p_hat, quiet = TRUE)
  
  auc_value <- as.numeric(auc(roc_obj))
  
  # The best threshold maximizes the criterion selected by `method`.
  best_cut <- coords(
    roc_obj,
    x = "best",
    best.method = method,
    ret = c("threshold", "sensitivity", "specificity", "accuracy"),
    transpose = FALSE
  )
  
  # If several thresholds tie, this coercion may require explicit tie handling.
  threshold_best <- as.numeric(best_cut$threshold)
  y_pred <- as.numeric(p_hat > threshold_best)
  
  # Confusion-matrix counts use AI-generated text as the positive class.
  TP <- sum(y_true == 1 & y_pred == 1)
  TN <- sum(y_true == 0 & y_pred == 0)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  
  # Return NA for undefined precision, recall, or F1 denominators.
  safe_div <- function(a, b) {
    ifelse(b == 0, NA, a / b)
  }
  
  accuracy <- safe_div(TP + TN, length(y_true))
  precision <- safe_div(TP, TP + FP)
  recall <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  F1 <- safe_div(2 * precision * recall, precision + recall)
  
  result <- data.frame(
    threshold = threshold_best,
    AUC = auc_value,
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1 = F1,
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN
  )
  
  return(list(
    result = result,
    roc = roc_obj,
    y_pred = y_pred
  ))
}




#' Plot fold-wise and mean ROC curves for three detection methods.
#'
#' Compares FIBER and FIBER-Stream posterior matrices with a full-length
#' Fast-DetectGPT score vector. Individual folds are dashed/translucent; mean
#' ROC curves are solid. The legend displays mean fold AUC and fold AUC SD.
#'
#' @param y_true Full-length binary label vector, ordered as the original data.
#' @param fold Full-length fold ID vector with integer labels 1, ..., `K`.
#' @param post_mat Matrix of held-out FIBER posterior probabilities. Column `k`
#'   must contain ONLY predictions for the rows where `fold == k`, in exactly the
#'   same order as `which(fold == k)`.
#' @param stream_mat Matrix of held-out FIBER-Stream posterior probabilities,
#'   using the same fold-column and row-order convention as `post_mat`.
#' @param score_vec Full-length Fast-DetectGPT score vector aligned with `y_true`.
#' @param K Number of cross-validation folds (default: 5).
#' @param score_direction pROC direction for Fast-DetectGPT (`"<"` means larger
#'   scores indicate the positive class).
#' @param legend_position Base R legend position.
#' @param main Optional figure title.
#' @param fpr_grid False-positive-rate interpolation grid for averaging ROC curves.
#' @return List with `post`, `stream`, and `score` components; each contains
#'   fold AUCs, fold true-positive-rate curves, and the mean TPR curve.
#' @note Load `pROC` before using this plot function (e.g., `library(pROC)`).
#'   This helper is specific to three methods; it does not accept arbitrary
#'   additional posterior matrices.

plot_cv_roc_three_methods <- function(
  y_true,
  fold,
  post_mat,
  stream_mat,
  score_vec,
  K = 5,
  score_direction = "<",
  legend_position="topleft",
  main="",
  fpr_grid = seq(0, 1, by = 0.01)
) {
  
  # Fold-specific posterior matrices are indexed by their fold columns.
  get_cv_roc_from_matrix <- function(Pmat) {
    
    auc_vec <- numeric(K)
    tpr_mat <- matrix(NA, nrow = length(fpr_grid), ncol = K)
    
    for (k in 1:K) {
      test_id <- which(fold == k)
      y_test <- y_true[test_id]
      # Assumes the rows in column k match which(fold == k) exactly.
      p_test <- Pmat[, k]
      
      roc_k <- roc(
        response = y_test,
        predictor = p_test,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
      
      auc_vec[k] <- as.numeric(auc(roc_k))
      
      tpr_mat[, k] <- as.numeric(coords(
        roc_k,
        x = 1 - fpr_grid,
        input = "specificity",
        ret = "sensitivity",
        transpose = FALSE
      )$sensitivity)
    }
    
    list(
      auc_vec = auc_vec,
      tpr_mat = tpr_mat,
      mean_tpr = rowMeans(tpr_mat, na.rm = TRUE)
    )
  }
  
  # Full-length score vectors are subset using the fold IDs.
  get_cv_roc_from_score <- function(score_input) {
    
    auc_vec <- numeric(K)
    tpr_mat <- matrix(NA, nrow = length(fpr_grid), ncol = K)
    
    for (k in 1:K) {
      test_id <- which(fold == k)
      y_test <- y_true[test_id]
      s_test <- score_input[test_id]
      
      roc_k <- roc(
        response = y_test,
        predictor = s_test,
        levels = c(0, 1),
        direction = score_direction,
        quiet = TRUE
      )
      
      auc_vec[k] <- as.numeric(auc(roc_k))
      
      tpr_mat[, k] <- as.numeric(coords(
        roc_k,
        x = 1 - fpr_grid,
        input = "specificity",
        ret = "sensitivity",
        transpose = FALSE
      )$sensitivity)
    }
    
    list(
      auc_vec = auc_vec,
      tpr_mat = tpr_mat,
      mean_tpr = rowMeans(tpr_mat, na.rm = TRUE)
    )
  }
  
  # Compute per-fold ROC curves before interpolating their mean TPR.
  res_post <- get_cv_roc_from_matrix(post_mat)
  res_stream <- get_cv_roc_from_matrix(stream_mat)
  res_score <- get_cv_roc_from_score(score_vec)
  
  plot(
    fpr_grid,
    res_post$mean_tpr,
    type = "n",
    xlim = c(0, 1),
    ylim = c(0, 1),
    xlab = "False Positive Rate",
    ylab = "True Positive Rate",
    main = main
  )
  
  # Diagonal reference line denotes chance discrimination.
  abline(a = 0, b = 1, lty = 3, col = "gray50")
  
  # Dashed translucent curves show each fold; solid lines show their mean.
  for (k in 1:K) {
    lines(fpr_grid, res_post$tpr_mat[, k],
          col = adjustcolor("red", alpha.f = 0.35),
          lty = 2, lwd = 2)
    
    lines(fpr_grid, res_stream$tpr_mat[, k],
          col = adjustcolor("orange", alpha.f = 0.35),
          lty = 2, lwd = 2)
    
    lines(fpr_grid, res_score$tpr_mat[, k],
          col = adjustcolor("blue", alpha.f = 0.35),
          lty = 2, lwd = 2)
  }
  
  lines(fpr_grid, res_post$mean_tpr, col = "red", lwd = 3)
  lines(fpr_grid, res_stream$mean_tpr, col = "orange", lwd = 3)
  lines(fpr_grid, res_score$mean_tpr, col = "blue", lwd = 3)
  
  # Legend lists mean AUC (SD of fold AUCs), not a pooled-data AUC.
  legend(
    legend_position,
    legend = c(
      paste0("FIBER: ", sprintf("%.3f", mean(res_post$auc_vec)),
             "(", sprintf("%.3f", sd(res_post$auc_vec)), ")"),
      paste0("FIBER-Stream: ", sprintf("%.3f", mean(res_stream$auc_vec)),
             "(", sprintf("%.3f", sd(res_stream$auc_vec)), ")"),
      paste0("Fast-DetectGPT: ", sprintf("%.3f", mean(res_score$auc_vec)),
             "(", sprintf("%.3f", sd(res_score$auc_vec)), ")")
    ),
    col = c("red", "orange", "blue"),
    lty = 1,
    lwd = 3,
    bty = "n"
  )
  
  return(list(
    post = res_post,
    stream = res_stream,
    score = res_score
  ))
}