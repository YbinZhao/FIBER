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
# EXPERIMENTAL IRT-MARGINAL VERSION (BOTH fiber() AND fiber_stream()).
# These functions replace the earlier independent Bernoulli vote likelihood
# with IRT-integrated class-conditional vote pattern likelihoods. Historical
# class proportions provide the shared prior; fitted IRT item distributions
# and detector abilities provide pattern-specific evidence at test time.
# Results are NOT interchangeable with the empirical-Bayes-prior-only version.
# A fitted IRT model (including both item vectors and detector abilities) is
# required but is not fitted by this script.
# =============================================================================

library(mvtnorm)

#' Evaluate the class-conditional likelihood of auxiliary detector outputs.
#'
#' Legacy independent-Bernoulli detector likelihood (comparison only).
#'
#' This helper is preserved for compatibility with older FIBER workflows.
#' The revised fiber() and fiber_stream() instead use the IRT-marginalized
#' likelihood from irt_detector_loglikelihood().
#'
#' @param p Positive-prediction probability for each detector.
#' @param D Binary detector-output matrix, with the same K detector columns.
#' @param eps Small clipping constant to prevent log(0) and 0 * (-Inf).
#' @return Numeric likelihood for each detector-output pattern.
detector_likelihood <- function(p, D, eps = 1e-10) {
  p <- as.numeric(p)
  D <- as.matrix(D)
  storage.mode(D) <- "double"
  if (length(p) != ncol(D) || any(!is.finite(D)) ||
      any(p < 0 | p > 1) || any(!is.finite(D)) ||
      any(!(D %in% c(0, 1))) ||
      length(eps) != 1L || !is.finite(eps) || eps <= 0 || eps >= 0.5) {
    stop("Invalid detector probabilities, binary outputs, or eps.")
  }
  p <- pmax(eps, pmin(1 - eps, p))
  log_h <- D %*% log(p) + (1 - D) %*% log1p(-p)
  as.numeric(exp(log_h))
}

#' Marginalize an IRT model over historical class-specific item representations.
#'
#' The fitted 2PL MIRT model specifies
#' q_k(z) = logistic(alpha' * (theta_k - beta)), where z=(alpha,beta).
#' This is the probability that detector k predicts the TRUE label correctly.
#' Thus, P(D_k=1 | z,Y=1)=q_k(z) and P(D_k=1 | z,Y=0)=1-q_k(z).
#' Given z and Y, detector outputs are assumed conditionally independent.
#' Integrating over f_y(z) induces dependence among detectors marginally.
#'
#' Monte Carlo integration uses samples z^(m) from a fitted Gaussian f_y and
#' computes log mean_m P(D | z^(m),Y=y) by log-sum-exp. Repeated binary output
#' patterns are evaluated only once and are processed in memory-limited batches.
#'
#' @param d_new Binary detector-output matrix for new texts (N x K).
#' @param theta_det Detector ability matrix (K x d), from the SAME fitted MIRT
#'   model and coordinate system as `z_train`.
#' @param mu_z Class-specific mean of the historical item vectors (length 2*d).
#' @param Sigma_z Class-specific covariance of the historical item vectors.
#' @param y Class label, 0 (human) or 1 (AI).
#' @param mc_samples Number of Gaussian draws used for Monte Carlo integration.
#' @param batch_size Maximum distinct detector patterns per matrix batch.
#' @param cov_ridge Nonnegative diagonal regularization for the covariance.
#' @param prob_clip Prevents log(0) when detector success probabilities saturate.
#' @return Log marginal detector-output likelihood for each new text.
irt_detector_loglikelihood <- function(
  d_new,
  theta_det,
  mu_z,
  Sigma_z,
  y,
  mc_samples = 2000L,
  batch_size = 256L,
  cov_ridge = 1e-6,
  prob_clip = 1e-10
) {
  d_new <- as.matrix(d_new)
  theta_det <- as.matrix(theta_det)
  storage.mode(d_new) <- "double"
  storage.mode(theta_det) <- "double"

  if (!(y %in% c(0, 1)) || length(y) != 1L) {
    stop("y must be exactly 0 or 1.")
  }
  if (ncol(theta_det) < 1L || nrow(theta_det) != ncol(d_new)) {
    stop("theta_det must have K rows, matching the detector columns in d_new.")
  }
  d <- ncol(theta_det)
  if (length(mu_z) != 2L * d ||
      !all(dim(Sigma_z) == c(2L * d, 2L * d))) {
    stop("The item vector must have 2*d columns ordered as alpha then beta.")
  }
  if (any(!is.finite(theta_det)) || any(!is.finite(mu_z)) ||
      any(!is.finite(Sigma_z)) || any(!is.finite(d_new)) ||
      any(!(d_new %in% c(0, 1)))) {
    stop("IRT parameters must be finite and detector outputs must be binary.")
  }
  if (!is.numeric(mc_samples) || length(mc_samples) != 1L ||
      !is.finite(mc_samples) || mc_samples < 1 ||
      mc_samples != floor(mc_samples) ||
      !is.numeric(batch_size) || length(batch_size) != 1L ||
      !is.finite(batch_size) || batch_size < 1 ||
      batch_size != floor(batch_size)) {
    stop("mc_samples and batch_size must be positive integers.")
  }
  if (!is.finite(cov_ridge) || cov_ridge < 0 ||
      !is.finite(prob_clip) || prob_clip <= 0 || prob_clip >= 0.5) {
    stop("Check cov_ridge and prob_clip.")
  }
  if (nrow(d_new) == 0L) return(numeric(0))

  Sigma_sample <- (Sigma_z + t(Sigma_z)) / 2 +
    diag(cov_ridge, nrow(Sigma_z))
  # A well-defined Gaussian density requires positive-definite covariance.
  if (inherits(try(chol(Sigma_sample), silent = TRUE), "try-error")) {
    stop("Class-specific IRT covariance is not positive definite; increase cov_ridge.")
  }

  z_draw <- mvtnorm::rmvnorm(
    n = as.integer(mc_samples), mean = mu_z, sigma = Sigma_sample
  )
  alpha_draw <- z_draw[, seq_len(d), drop = FALSE]
  beta_draw  <- z_draw[, d + seq_len(d), drop = FALSE]

  # M x K matrix: P(correct prediction | sampled item and detector).
  logits <- alpha_draw %*% t(theta_det) -
    rowSums(alpha_draw * beta_draw)
  q_correct <- plogis(logits)

  # Translate correctness into the probability of an AI (=1) detector vote.
  r <- if (y == 1L) q_correct else 1 - q_correct

  r[r < prob_clip] <- prob_clip
  r[r > 1 - prob_clip] <- 1 - prob_clip

  log_r <- log(r)
  log_one_minus_r <- log1p(-r)
  
  # P(D|z_m,y) = exp(sum_k log(1-r_mk) + D_k*logit(r_mk)).
  log_ratio <- log_r - log_one_minus_r       # M x K
  offsets <- rowSums(log_one_minus_r)       # M

  # Cache observed patterns: for K=13 there are at most 2^13 possibilities.
  keys <- apply(d_new, 1L, paste0, collapse = "")
  keep <- !duplicated(keys)
  patterns <- d_new[keep, , drop = FALSE]
  mapping <- match(keys, keys[keep])
  n_patterns <- nrow(patterns)
  unique_log_lik <- numeric(n_patterns)

  for (first in seq.int(1L, n_patterns, by = as.integer(batch_size))) {
    last <- min(first + as.integer(batch_size) - 1L, n_patterns)
    rows <- first:last

    # Each column is one Monte Carlo draw and each row one observed pattern.
    log_cond <- patterns[rows, , drop = FALSE] %*% t(log_ratio)
    log_cond <- sweep(log_cond, 2L, offsets, "+")

    # Stable log(mean(exp(log P(D|z_m,y)))) over the M sampled items.
    max_log <- apply(log_cond, 1L, max)
    unique_log_lik[rows] <- max_log +
      log(rowMeans(exp(sweep(log_cond, 1L, max_log, "-"))))
  }

  unique_log_lik[mapping]
}

#' Evaluate an expression under a reproducible seed without changing the
#' caller's existing global random-number generator state.
with_preserved_seed <- function(seed, expr) {
  if (is.null(seed)) return(eval(substitute(expr), parent.frame()))
  if (!is.numeric(seed) || length(seed) != 1L ||
      !is.finite(seed) || seed != floor(seed) ||
      abs(seed) > .Machine$integer.max) {
    stop("mc_seed must be NULL or a finite integer supported by set.seed().")
  }
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) previous_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", previous_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))
  eval(substitute(expr), parent.frame())
}

#' Apply FIBER with an IRT-marginalized detector-output likelihood.
#'
#' This version replaces the original independent Bernoulli detector likelihood
#' with h_y(D) = E_{Z|Y=y}[P(D|Z,Y=y)], evaluated by Monte Carlo integration.
#' The shared prior is now the observed historical AI-class proportion; IRT
#' influences TEST-SPECIFIC evidence through the output pattern D, not through a
#' shared IRT-informed prior. The score density remains class-conditional Gaussian.
#'
#' @param s_train Continuous Fast-DetectGPT scores of historical texts.
#' @param y_train Historical labels: 0=human, 1=AI.
#' @param d_train Historical detector-output matrix (N_train x K). It is used
#'   to verify alignment; its correctness matrix must already have been used
#'   to fit the IRT model that produced z_train and theta_det.
#' @param z_train Historical IRT item-parameter matrix (N_train x 2*d), ordered
#'   as [alpha_1,...,alpha_d,beta_1,...,beta_d]. Rows match y_train.
#' @param s_test Continuous Fast-DetectGPT scores for incoming texts.
#' @param d_test Incoming 0/1 detector-output matrix (N_test x K).
#' @param w Evidence exponents for prior, score, and IRT detector likelihood.
#' @param a_0,b_0 Retained for compatibility with older calls; not used to
#'   construct the empirical class-proportion prior in this version.
#' @param threshold Posterior classification threshold (0 <= threshold <= 1).
#' @param theta_det REQUIRED detector ability matrix (K x d) from the fitted
#'   2PL MIRT model, with row order matching d_train/d_test columns.
#' @param mc_samples Number of Monte Carlo draws PER class (default 2000).
#' @param mc_seed Seed for reproducible Monte Carlo. NULL uses global RNG.
#' @param batch_size Number of unique detector patterns processed at once.
#' @param cov_ridge Diagonal covariance stabilization for Gaussian item draws.
#' @param score_variance_floor Small positive floor for score variances.
#' @return List containing posterior, y_pred, pi_prior, a deprecated pi_irt
#'   alias, w, estimated model parameters, and detector likelihoods.
#' @note fiber_stream() uses the same IRT-integrated detector likelihood,
#'   but adapts the score distribution through unlabeled-batch EM.
#' @note Use parameters and training-fold-only thresholds when evaluating on
#'   held-out data. No labels for new texts enter this likelihood calculation.
fiber <- function(
  s_train,
  y_train,
  d_train,
  z_train,
  s_test,
  d_test,
  w = c(1, 1, 1),
  a_0 = 1,
  b_0 = 1,
  threshold = 0.5,
  theta_det = NULL,
  mc_samples = 2000L,
  mc_seed = 20261028L,
  batch_size = 256L,
  cov_ridge = 1e-6,
  score_variance_floor = 1e-8
) {
  if (is.null(theta_det)) {
    stop("theta_det is required: provide the K x d detector abilities from the fitted IRT model.")
  }
  s_train <- as.numeric(s_train)
  y_train <- as.numeric(y_train)
  d_train <- as.matrix(d_train)
  z_train <- as.matrix(z_train)
  s_test <- as.numeric(s_test)
  d_test <- as.matrix(d_test)
  theta_det <- as.matrix(theta_det)
  storage.mode(d_train) <- "double"
  storage.mode(z_train) <- "double"
  storage.mode(d_test) <- "double"
  storage.mode(theta_det) <- "double"

  if (length(y_train) != length(s_train) ||
      length(y_train) != nrow(d_train) ||
      length(y_train) != nrow(z_train) ||
      length(s_test) != nrow(d_test)) {
    stop("Historical and incoming scores, labels, and matrices must have matching rows.")
  }
  if (any(!is.finite(s_train)) || any(!is.finite(s_test)) ||
      any(!is.finite(z_train)) || any(!is.finite(theta_det)) ||
      any(!is.finite(d_train)) || any(!is.finite(d_test))) {
    stop("Inputs must be finite; handle missing values before calling fiber().")
  }
  if (any(!(y_train %in% c(0, 1))) ||
      any(!(d_train %in% c(0, 1))) || any(!(d_test %in% c(0, 1)))) {
    stop("Labels and detector outputs must contain only 0 and 1.")
  }
  if (ncol(d_train) != ncol(d_test) ||
      ncol(d_train) != nrow(theta_det) ||
      ncol(z_train) != 2L * ncol(theta_det)) {
    stop("Dimensions must satisfy: D columns=theta rows=K; Z columns=2*theta columns.")
  }
  if (!is.null(colnames(d_train)) && !is.null(colnames(d_test)) &&
      !identical(colnames(d_train), colnames(d_test))) {
    stop("The detector columns in d_train and d_test must be in the same order.")
  }
  if (!is.null(colnames(d_train)) && !is.null(rownames(theta_det)) &&
      !identical(colnames(d_train), rownames(theta_det))) {
    stop("theta_det rows must match the detector column names and order.")
  }
  if (length(w) != 3L || any(!is.finite(w)) || any(w < 0)) {
    stop("w must be three finite, nonnegative evidence weights.")
  }
  if (length(threshold) != 1L || !is.numeric(threshold) ||
      !is.finite(threshold) || threshold < 0 || threshold > 1) {
    stop("threshold must lie in [0, 1].")
  }
  if (!is.finite(score_variance_floor) || score_variance_floor <= 0) {
    stop("score_variance_floor must be positive.")
  }
  if (!isTRUE(all.equal(a_0, 1)) || !isTRUE(all.equal(b_0, 1))) {
    warning("a_0 and b_0 are deprecated and not used by the empirical-prior IRT-likelihood model.")
  }

  idx_A <- which(y_train == 1)
  idx_H <- which(y_train == 0)
  n_A <- length(idx_A)
  n_H <- length(idx_H)
  if (n_A < 2L || n_H < 2L) {
    stop("At least two historical texts are needed in each class.")
  }

  # Estimate Gaussian distributions for historical item representations.
  mu_A_z <- colMeans(z_train[idx_A, , drop = FALSE])
  mu_H_z <- colMeans(z_train[idx_H, , drop = FALSE])
  Sigma_A_z <- cov(z_train[idx_A, , drop = FALSE])
  Sigma_H_z <- cov(z_train[idx_H, , drop = FALSE])

  # Empirical class prior: the original IRT-soft-count prior is NOT used.
  pi_prior <- n_A / length(y_train)

  # Estimate historical class-conditional Fast-DetectGPT score densities.
  mu_A_s <- mean(s_train[idx_A])
  mu_H_s <- mean(s_train[idx_H])
  sigma2_A_s <- max(var(s_train[idx_A]), score_variance_floor)
  sigma2_H_s <- max(var(s_train[idx_H]), score_variance_floor)
  log_g_A_s <- dnorm(s_test, mean = mu_A_s,
                     sd = sqrt(sigma2_A_s), log = TRUE)
  log_g_H_s <- dnorm(s_test, mean = mu_H_s,
                     sd = sqrt(sigma2_H_s), log = TRUE)

  # Integrate detector-output pattern likelihoods over f_y(Z) using a
  # reproducible Monte Carlo stream; both classes use different draws.
  log_h <- with_preserved_seed(mc_seed, {
    list(
      A = irt_detector_loglikelihood(
        d_new = d_test, theta_det = theta_det,
        mu_z = mu_A_z, Sigma_z = Sigma_A_z, y = 1L,
        mc_samples = mc_samples, batch_size = batch_size,
        cov_ridge = cov_ridge
      ),
      H = irt_detector_loglikelihood(
        d_new = d_test, theta_det = theta_det,
        mu_z = mu_H_z, Sigma_z = Sigma_H_z, y = 0L,
        mc_samples = mc_samples, batch_size = batch_size,
        cov_ridge = cov_ridge
      )
    )
  })
  log_h_A_d <- log_h$A
  log_h_H_d <- log_h$H

  # Posterior in log-odds form to avoid multiplying tiny likelihood values.
  log_odds <- w[1] * qlogis(pi_prior) +
    w[2] * (log_g_A_s - log_g_H_s) +
    w[3] * (log_h_A_d - log_h_H_d)
  posterior <- plogis(log_odds)
  y_pred <- as.numeric(posterior > threshold)

  return(list(
    posterior = posterior,
    y_pred = y_pred,
    pi_prior = pi_prior,
    pi_irt = pi_prior,  # Deprecated compatibility alias: not an IRT-based prior.
    eta = NULL,         # No historical IRT soft counts in the revised model.
    w = w,
    parameters = list(
      Z = list(mu_A_z = mu_A_z, mu_H_z = mu_H_z,
               Sigma_A_z = Sigma_A_z, Sigma_H_z = Sigma_H_z),
      s = list(mu_A_s = mu_A_s, mu_H_s = mu_H_s,
               sigma2_A_s = sigma2_A_s, sigma2_H_s = sigma2_H_s),
      detector = list(theta_det = theta_det,
                      likelihood = "IRT marginal Monte Carlo",
                      mc_samples = mc_samples, cov_ridge = cov_ridge)
    ),
    likelihoods = list(
      g_A_s = exp(log_g_A_s), g_H_s = exp(log_g_H_s),
      h_A_d = exp(log_h_A_d), h_H_d = exp(log_h_H_d),
      log_g_A_s = log_g_A_s, log_g_H_s = log_g_H_s,
      log_h_A_d = log_h_A_d, log_h_H_d = log_h_H_d
    )
  ))
}

#' Perform a constrained Gaussian M-step from soft class memberships.
#'
#' The two components are assigned to machine (A) and human (H) classes by
#' their fixed detector-output likelihoods, so their identities must NOT be
#' swapped during EM. If unconstrained score means reverse their expected
#' ordering (AI >= human), optimize the common-mean boundary instead.
#' `sigma_as` and `sigma_hs` are variances, not standard deviations.
fiber_gaussian_mstep <- function(s, w, variance_floor = 1e-8) {
  n_A <- sum(w)
  n_H <- sum(1 - w)
  if (n_A <= 1e-10 || n_H <= 1e-10) {
    stop("An EM class has nearly zero effective sample size.")
  }
  mu_A <- sum(w * s) / n_A
  mu_H <- sum((1 - w) * s) / n_H
  sigma_A <- max(sum(w * (s - mu_A)^2) / n_A, variance_floor)
  sigma_H <- max(sum((1 - w) * (s - mu_H)^2) / n_H, variance_floor)

  # The usual unconstrained weighted-Gaussian M-step.
  if (mu_A >= mu_H) {
    return(list(mu_as = mu_A, mu_hs = mu_H,
                sigma_as = sigma_A, sigma_hs = sigma_H,
                orientation_boundary = FALSE))
  }

  # Constrained maximum lies on mu_A = mu_H when the unconstrained means
  # violate the requested score orientation. Optimize the boundary likelihood
  # with class-specific variances (subject to their positive variance floors).
  boundary_Q <- function(mu) {
    res_A <- sum(w * (s - mu)^2)
    res_H <- sum((1 - w) * (s - mu)^2)
    var_A <- max(res_A / n_A, variance_floor)
    var_H <- max(res_H / n_H, variance_floor)
    -0.5 * (n_A * log(2 * pi * var_A) + res_A / var_A +
              n_H * log(2 * pi * var_H) + res_H / var_H)
  }
  if (min(s) == max(s)) {
    common_mean <- s[1L]
  } else {
    common_mean <- optimize(boundary_Q, interval = range(s),
                            maximum = TRUE)$maximum
  }
  sigma_A <- max(sum(w * (s - common_mean)^2) / n_A, variance_floor)
  sigma_H <- max(sum((1 - w) * (s - common_mean)^2) / n_H, variance_floor)
  list(mu_as = common_mean, mu_hs = common_mean,
       sigma_as = sigma_A, sigma_hs = sigma_H,
       orientation_boundary = TRUE)
}

#' Fit class-conditional Fast-DetectGPT score distributions from unlabeled data.
#'
#' Uses a fixed historical AI-class prior and FIXED IRT-marginal detector-output
#' log likelihoods. Only the two Gaussian score components are updated via EM.
#' For backward compatibility, ordinary likelihoods may be passed as h_ad/h_hd
#' when `input_log=FALSE`; the revised FIBER-Stream passes logarithms and sets
#' `input_log=TRUE` for numerical stability.
#'
#' @param w0 Initial class-A responsibilities based on detector evidence.
#' @param s Numeric vector of unlabeled Fast-DetectGPT scores.
#' @param h_ad,h_hd Class-conditional detector likelihoods (or log-likelihoods).
#' @param pi_irt Historical class prior. Name preserved for old callers; the
#'   revised method uses the empirical class proportion, not IRT soft counts.
#' @param max_iter Maximum EM iterations.
#' @param tol Absolute observed-data log-likelihood change for stopping.
#' @param verbose Print iteration progress.
#' @param input_log Whether h_ad and h_hd are already log likelihoods.
#' @param score_variance_floor Positive lower bound for Gaussian variances.
#' @return List containing class means and variances, final responsibilities,
#'   estimated parameter list (`params`), log-likelihood trace, and convergence.
fiber_stream_em <- function(
  w0, s, h_ad, h_hd, pi_irt,
  max_iter = 500, tol = 1e-6, verbose = FALSE,
  input_log = FALSE, score_variance_floor = 1e-8
) {
  s <- as.numeric(s)
  w0 <- as.numeric(w0)
  h_ad <- as.numeric(h_ad)
  h_hd <- as.numeric(h_hd)
  n <- length(s)
  if (n < 2L || length(w0) != n || length(h_ad) != n ||
      length(h_hd) != n || any(!is.finite(s)) ||
      any(!is.finite(w0)) || any(w0 < 0 | w0 > 1)) {
    stop("EM input lengths must match; scores/responsibilities must be finite.")
  }
  if (length(pi_irt) != 1L || !is.finite(pi_irt) ||
      pi_irt <= 0 || pi_irt >= 1) {
    stop("pi_irt must be a scalar prior strictly between 0 and 1.")
  }
  if (length(max_iter) != 1L || !is.finite(max_iter) ||
      max_iter < 1L || max_iter != floor(max_iter) ||
      length(tol) != 1L || !is.finite(tol) || tol <= 0 ||
      length(score_variance_floor) != 1L ||
      !is.finite(score_variance_floor) || score_variance_floor <= 0) {
    stop("Check max_iter, tol and score_variance_floor.")
  }
  if (input_log) {
    log_h_A <- h_ad
    log_h_H <- h_hd
    if (any(!is.finite(log_h_A)) || any(!is.finite(log_h_H))) {
      stop("Log detector likelihoods must be finite.")
    }
  } else {
    if (any(!is.finite(h_ad)) || any(!is.finite(h_hd)) ||
        any(h_ad < 0 | h_hd < 0)) {
      stop("Detector likelihoods must be nonnegative and finite.")
    }
    log_h_A <- log(pmax(h_ad, .Machine$double.xmin))
    log_h_H <- log(pmax(h_hd, .Machine$double.xmin))
  }

  log_prior_A <- log(pi_irt)
  log_prior_H <- log1p(-pi_irt)
  params <- fiber_gaussian_mstep(s, w0,
                                variance_floor = score_variance_floor)
  # Stable two-component mixture log likelihood with fixed detector evidence.
  score_loglik <- function(par) {
    log_A <- log_prior_A + log_h_A +
      dnorm(s, mean = par$mu_as, sd = sqrt(par$sigma_as), log = TRUE)
    log_H <- log_prior_H + log_h_H +
      dnorm(s, mean = par$mu_hs, sd = sqrt(par$sigma_hs), log = TRUE)
    max_log <- pmax(log_A, log_H)
    sum(max_log + log(exp(log_A - max_log) + exp(log_H - max_log)))
  }
  loglik_old <- score_loglik(params)
  trace <- data.frame(
    iter = integer(), loglik = numeric(), diff = numeric(),
    mu_as = numeric(), sigma_as = numeric(),
    mu_hs = numeric(), sigma_hs = numeric(),
    orientation_boundary = logical()
  )
  converged <- FALSE

  for (iter in seq_len(max_iter)) {
    log_A <- log_prior_A + log_h_A +
      dnorm(s, mean = params$mu_as, sd = sqrt(params$sigma_as), log = TRUE)
    log_H <- log_prior_H + log_h_H +
      dnorm(s, mean = params$mu_hs, sd = sqrt(params$sigma_hs), log = TRUE)
    responsibilities <- plogis(log_A - log_H)
    params_new <- fiber_gaussian_mstep(
      s, responsibilities, variance_floor = score_variance_floor
    )
    loglik <- score_loglik(params_new)
    difference <- abs(loglik - loglik_old)
    trace <- rbind(trace, data.frame(
      iter = iter, loglik = loglik, diff = difference,
      mu_as = params_new$mu_as, sigma_as = params_new$sigma_as,
      mu_hs = params_new$mu_hs, sigma_hs = params_new$sigma_hs,
      orientation_boundary = params_new$orientation_boundary
    ))
    params <- params_new
    if (verbose && (iter %% 10L == 0L || difference <= tol)) {
      cat("Iter:", iter, "loglik:", round(loglik, 6),
          "diff:", signif(difference, 4), "\n")
    }
    if (difference <= tol) {
      converged <- TRUE
      break
    }
    loglik_old <- loglik
  }

  # Re-evaluate posterior responsibilities using the FINAL EM parameters.
  log_A <- log_prior_A + log_h_A +
    dnorm(s, mean = params$mu_as, sd = sqrt(params$sigma_as), log = TRUE)
  log_H <- log_prior_H + log_h_H +
    dnorm(s, mean = params$mu_hs, sd = sqrt(params$sigma_hs), log = TRUE)
  w_final <- plogis(log_A - log_H)
  list(
    mu_as = params$mu_as, sigma_as = params$sigma_as,
    mu_hs = params$mu_hs, sigma_hs = params$sigma_hs,
    w = w_final,
    params = params,
    trace = trace,
    loglik = tail(trace$loglik, 1L),
    iterations = nrow(trace),
    converged = converged
  )
}

#' Apply FIBER-Stream with an IRT-marginalized detector-output likelihood.
#'
#' The historical class-conditional Gaussian distributions of IRT item
#' representations and detector abilities define marginal detector-output
#' likelihoods through Monte Carlo integration. These likelihoods and an
#' empirical class prior remain fixed; only the Gaussian Fast-DetectGPT score
#' distributions are adapted from the unlabeled incoming batch by EM.
#'
#' @param y_train Historical binary labels (0=human, 1=AI).
#' @param D_train Historical auxiliary detector-output matrix (N_train x K).
#' @param Z_train Historical item vectors (N_train x 2*d), ordered alpha then beta.
#' @param s_stream Fast-DetectGPT score for each incoming text (N_stream).
#' @param D_stream Incoming detector outputs (N_stream x K).
#' @param pi_0,a_0,b_0 Deprecated placeholders retained for old call syntax;
#'   they are not used in the empirical-class-proportion prior.
#' @param threshold Posterior classification threshold.
#' @param max_iter Maximum number of EM iterations.
#' @param verbose Whether to show EM progress.
#' @param theta_det REQUIRED detector ability matrix (K x d), matching the
#'   same fitted IRT model and detector ordering as Z_train and D_train.
#' @param mc_samples Number of Monte Carlo samples per class.
#' @param mc_seed Local RNG seed; NULL uses the caller's RNG.
#' @param batch_size Number of distinct detector patterns evaluated per batch.
#' @param cov_ridge Diagonal covariance regularization for IRT sampling.
#' @param score_variance_floor Positive lower bound for score variances in EM.
#' @param tol Absolute observed log-likelihood change for EM stopping.
#' @return List with posterior, y_pred, pi_prior (pi_irt alias), omega_init,
#'   fitted EM parameters, and IRT-marginal detector log likelihoods.
#' @note This is BATCHWISE unsupervised score adaptation, not online learning
#'   with per-observation recursive parameter updates.
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
  verbose = FALSE,
  theta_det = NULL,
  mc_samples = 2000L,
  mc_seed = 20261028L,
  batch_size = 256L,
  cov_ridge = 1e-6,
  score_variance_floor = 1e-8,
  tol = 1e-6
) {
  if (is.null(theta_det)) {
    stop("theta_det is required: provide the K x d abilities from the fitted IRT model.")
  }
  y_train <- as.numeric(y_train)
  D_train <- as.matrix(D_train)
  Z_train <- as.matrix(Z_train)
  s_stream <- as.numeric(s_stream)
  D_stream <- as.matrix(D_stream)
  theta_det <- as.matrix(theta_det)
  storage.mode(D_train) <- "double"
  storage.mode(Z_train) <- "double"
  storage.mode(D_stream) <- "double"
  storage.mode(theta_det) <- "double"

  if (length(y_train) != nrow(D_train) ||
      length(y_train) != nrow(Z_train) ||
      length(s_stream) != nrow(D_stream) || length(s_stream) < 2L) {
    stop("Historical labels and incoming scores must align with their matrices.")
  }
  if (any(!is.finite(y_train)) || any(!is.finite(D_train)) ||
      any(!is.finite(Z_train)) || any(!is.finite(s_stream)) ||
      any(!is.finite(D_stream)) || any(!is.finite(theta_det)) ||
      any(!(y_train %in% c(0, 1))) ||
      any(!(D_train %in% c(0, 1))) ||
      any(!(D_stream %in% c(0, 1)))) {
    stop("All inputs must be finite and labels/detector outputs must be binary.")
  }
  if (ncol(D_train) != ncol(D_stream) ||
      ncol(D_train) != nrow(theta_det) ||
      ncol(Z_train) != 2L * ncol(theta_det)) {
    stop("Dimensions require K detector columns, K x d abilities, and 2*d item columns.")
  }
  if (!is.null(colnames(D_train)) && !is.null(colnames(D_stream)) &&
      !identical(colnames(D_train), colnames(D_stream))) {
    stop("Detector columns in D_train and D_stream must have the same order.")
  }
  if (!is.null(colnames(D_train)) && !is.null(rownames(theta_det)) &&
      !identical(colnames(D_train), rownames(theta_det))) {
    stop("theta_det rows must match historical detector column names.")
  }
  if (length(threshold) != 1L || !is.numeric(threshold) ||
      !is.finite(threshold) || threshold < 0 || threshold > 1) {
    stop("threshold must lie in [0,1].")
  }
  if (!isTRUE(all.equal(pi_0, 0.5)) ||
      !isTRUE(all.equal(a_0, 1)) || !isTRUE(all.equal(b_0, 1))) {
    warning("pi_0, a_0 and b_0 are deprecated in the IRT-marginal likelihood model.")
  }
  idx_A <- which(y_train == 1)
  idx_H <- which(y_train == 0)
  n_A <- length(idx_A)
  n_H <- length(idx_H)
  if (n_A < 2L || n_H < 2L) {
    stop("At least two historical samples per class are required.")
  }
  pi_prior <- n_A / length(y_train)

  mu_A_z <- colMeans(Z_train[idx_A, , drop = FALSE])
  mu_H_z <- colMeans(Z_train[idx_H, , drop = FALSE])
  Sigma_A_z <- cov(Z_train[idx_A, , drop = FALSE])
  Sigma_H_z <- cov(Z_train[idx_H, , drop = FALSE])

  # Same seeded integration used in standard FIBER: no test labels enter.
  log_h <- with_preserved_seed(mc_seed, {
    list(
      A = irt_detector_loglikelihood(
        d_new = D_stream, theta_det = theta_det,
        mu_z = mu_A_z, Sigma_z = Sigma_A_z, y = 1L,
        mc_samples = mc_samples, batch_size = batch_size,
        cov_ridge = cov_ridge
      ),
      H = irt_detector_loglikelihood(
        d_new = D_stream, theta_det = theta_det,
        mu_z = mu_H_z, Sigma_z = Sigma_H_z, y = 0L,
        mc_samples = mc_samples, batch_size = batch_size,
        cov_ridge = cov_ridge
      )
    )
  })
  log_h_A <- log_h$A
  log_h_H <- log_h$H

  # Detector-only responsibilities initialize the unlabeled-batch EM.
  omega_init <- plogis(qlogis(pi_prior) + log_h_A - log_h_H)
  em_fit <- fiber_stream_em(
    w0 = omega_init, s = s_stream,
    h_ad = log_h_A, h_hd = log_h_H,
    pi_irt = pi_prior,
    max_iter = max_iter, tol = tol, verbose = verbose,
    input_log = TRUE,
    score_variance_floor = score_variance_floor
  )

  # Full posterior with EM-adapted score densities and IRT detector evidence.
  log_g_A <- dnorm(s_stream, mean = em_fit$mu_as,
                   sd = sqrt(em_fit$sigma_as), log = TRUE)
  log_g_H <- dnorm(s_stream, mean = em_fit$mu_hs,
                   sd = sqrt(em_fit$sigma_hs), log = TRUE)
  posterior <- plogis(qlogis(pi_prior) + log_g_A - log_g_H +
                       log_h_A - log_h_H)
  y_pred <- as.numeric(posterior > threshold)

  list(
    posterior = posterior,
    y_pred = y_pred,
    pi_prior = pi_prior,
    pi_irt = pi_prior,   # Deprecated compatibility alias, not an IRT soft prior.
    eta = NULL,
    omega_init = omega_init,
    em = em_fit,
    parameters = list(
      IRT = list(mu_A_Z = mu_A_z, mu_H_Z = mu_H_z,
                 Sigma_A_Z = Sigma_A_z, Sigma_H_Z = Sigma_H_z),
      detector = list(theta_det = theta_det,
                      likelihood = "IRT marginal Monte Carlo",
                      mc_samples = mc_samples,
                      cov_ridge = cov_ridge),
      score = em_fit$params
    ),
    likelihoods = list(
      h_A = exp(log_h_A), h_H = exp(log_h_H),
      log_h_A = log_h_A, log_h_H = log_h_H,
      g_A = exp(log_g_A), g_H = exp(log_g_H),
      log_g_A = log_g_A, log_g_H = log_g_H
    )
  )
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
#' @param threshold Optional preselected cutoff (e.g., chosen on training data).
#'   When NULL, chooses the cutoff on the supplied labels; do not use test
#'   labels to select an operating cutoff for held-out evaluation.
#' @return List with a one-row metric table `result`, the ROC object `roc`, and
#'   the binary prediction vector `y_pred`.
#' @note This routine SELECTS and EVALUATES a cutoff on the SAME labels.
#'   For unbiased held-out evaluation, select the cutoff using training-fold
#'   predictions and labels, then apply it to the test scores without using
#'   test labels to select the cutoff. Multiple tied optimal thresholds may
#'   require choosing one explicitly.

eval_with_best_threshold <- function(y_true, p_hat, method = "youden",
                                     threshold = NULL) {
  
  library(pROC)
  
  y_true <- as.numeric(y_true)
  p_hat <- as.numeric(p_hat)
  
  # ROC and threshold use the labels supplied to THIS function.
  roc_obj <- pROC::roc(response = y_true, predictor = p_hat,
                       levels = c(0, 1), direction = "<", quiet = TRUE)
  
  auc_value <- as.numeric(pROC::auc(roc_obj))
  
  if (is.null(threshold)) {
    # Fit the operating threshold on the supplied labels; use TRAINING data.
    best_cut <- pROC::coords(
      roc_obj, x = "best", best.method = method,
      ret = "threshold", transpose = FALSE
    )
    # Multiple Youden-optimal thresholds are possible; use the first.
    threshold_best <- as.numeric(best_cut$threshold[1L])
  } else {
    threshold_best <- as.numeric(threshold)
    if (length(threshold_best) != 1L || !is.finite(threshold_best)) {
      stop("threshold must be a finite scalar.")
    }
  }
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
  
  if (length(y_true) != length(fold) || length(score_vec) != length(y_true) ||
      length(unique(fold)) != K || !setequal(unique(fold), seq_len(K))) {
    stop("y_true, fold, and score_vec must be aligned; folds must be 1,...,K.")
  }
  post_mat <- as.matrix(post_mat)
  stream_mat <- as.matrix(stream_mat)
  if (ncol(post_mat) != K || ncol(stream_mat) != K ||
      any(tabulate(fold, nbins = K) != nrow(post_mat)) ||
      any(tabulate(fold, nbins = K) != nrow(stream_mat))) {
    stop("Posterior matrices must have one column per fold and rows per test fold.")
  }
  # Fold-specific posterior matrices are indexed by their fold columns.
  get_cv_roc_from_matrix <- function(Pmat) {
    
    auc_vec <- numeric(K)
    tpr_mat <- matrix(NA, nrow = length(fpr_grid), ncol = K)
    
    for (k in 1:K) {
      test_id <- which(fold == k)
      y_test <- y_true[test_id]
      # Assumes the rows in column k match which(fold == k) exactly.
      p_test <- Pmat[, k]
      
      roc_k <- pROC::roc(
        response = y_test,
        predictor = p_test,
        levels = c(0, 1),
        direction = "<",
        quiet = TRUE
      )
      
      auc_vec[k] <- as.numeric(pROC::auc(roc_k))
      
      tpr_mat[, k] <- as.numeric(pROC::coords(
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
      
      roc_k <- pROC::roc(
        response = y_test,
        predictor = s_test,
        levels = c(0, 1),
        direction = score_direction,
        quiet = TRUE
      )
      
      auc_vec[k] <- as.numeric(pROC::auc(roc_k))
      
      tpr_mat[, k] <- as.numeric(pROC::coords(
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
