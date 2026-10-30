# Synthetic checks for BOTH FIBER and FIBER-Stream with IRT-integrated vote likelihoods.
# Run in the directory containing Fiber_IRT_full.R:
#   install.packages(c("mvtnorm", "pROC"))
#   source("Fiber_IRT_full_smoke_test.R")
# No real-world detector outputs or true test labels are needed by the inference functions.
source("Fiber_documented.R")
set.seed(123)
K <- 4L
latent_dim <- 2L
n_train <- 240L
n_test <- 80L

y_train <- rep(c(0, 1), each = n_train / 2L)
y_test <- rep(c(0, 1), each = n_test / 2L)
theta_det <- matrix(rnorm(K * latent_dim, sd = 0.65), nrow = K)
rownames(theta_det) <- paste0("detector_", seq_len(K))

z_train <- matrix(rnorm(n_train * 2L * latent_dim), nrow = n_train)
z_test <- matrix(rnorm(n_test * 2L * latent_dim), nrow = n_test)
# Only historical representations are passed to inference; z_test is
# used here solely to simulate detector votes for the synthetic example.
z_train[y_train == 1, 1] <- z_train[y_train == 1, 1] + 0.4
z_test[y_test == 1, 1] <- z_test[y_test == 1, 1] + 0.4

simulate_detector_votes <- function(z, y, theta) {
  d <- ncol(theta)
  a <- z[, seq_len(d), drop = FALSE]
  b <- z[, d + seq_len(d), drop = FALSE]
  logit_q <- a %*% t(theta) - rowSums(a * b)
  q <- plogis(logit_q)
  r <- q
  r[y == 0, ] <- 1 - q[y == 0, , drop = FALSE]
  D <- matrix(rbinom(length(r), 1L, prob = as.vector(r)), nrow = nrow(r))
  colnames(D) <- rownames(theta)
  D
}

d_train <- simulate_detector_votes(z_train, y_train, theta_det)
d_test <- simulate_detector_votes(z_test, y_test, theta_det)
s_train <- rnorm(n_train, mean = ifelse(y_train == 1, 0.7, -0.4), sd = 0.9)
s_test <- rnorm(n_test, mean = ifelse(y_test == 1, 0.7, -0.4), sd = 0.9)

seed_before <- .Random.seed
fit <- fiber(
  s_train = s_train, y_train = y_train,
  d_train = d_train, z_train = z_train,
  s_test = s_test, d_test = d_test,
  theta_det = theta_det,
  mc_samples = 500L, mc_seed = 20261028L
)
stopifnot(identical(seed_before, .Random.seed))

seed_before <- .Random.seed
fit_stream <- fiber_stream(
  y_train = y_train, D_train = d_train, Z_train = z_train,
  s_stream = s_test, D_stream = d_test,
  theta_det = theta_det,
  mc_samples = 500L, mc_seed = 20261028L,
  max_iter = 100L
)
stopifnot(identical(seed_before, .Random.seed))

for (res in list(fit, fit_stream)) {
  stopifnot(
    length(res$posterior) == n_test,
    all(is.finite(res$posterior)),
    all(res$posterior >= 0 & res$posterior <= 1),
    length(res$y_pred) == n_test,
    is.finite(res$pi_prior),
    res$pi_prior == mean(y_train)
  )
}

# Both models must use exactly the same class-conditional detector likelihoods
# when supplied with the same historical IRT model, incoming patterns, and seed.
stopifnot(
  isTRUE(all.equal(fit$likelihoods$log_h_A_d,
                   fit_stream$likelihoods$log_h_A, tolerance = 1e-12)),
  isTRUE(all.equal(fit$likelihoods$log_h_H_d,
                   fit_stream$likelihoods$log_h_H, tolerance = 1e-12)),
  is.list(fit_stream$parameters$score),
  fit_stream$em$mu_as >= fit_stream$em$mu_hs,
  all(is.finite(fit_stream$em$trace$loglik)),
  length(fit_stream$em$w) == n_test
)

# Integrated likelihoods over ALL possible K-bit vote patterns must sum to one.
patterns <- as.matrix(expand.grid(rep(list(0:1), K)))
for (class_y in c(0L, 1L)) {
  sub_z <- z_train[y_train == class_y, , drop = FALSE]
  log_h <- irt_detector_loglikelihood(
    d_new = patterns, theta_det = theta_det,
    mu_z = colMeans(sub_z), Sigma_z = cov(sub_z), y = class_y,
    mc_samples = 500L
  )
  stopifnot(abs(sum(exp(log_h)) - 1) < 1e-8)
}

# The threshold can be selected on training predictions and applied unchanged.
if (requireNamespace("pROC", quietly = TRUE)) {
  train_predictions <- fiber(
    s_train = s_train, y_train = y_train,
    d_train = d_train, z_train = z_train,
    s_test = s_train, d_test = d_train,
    theta_det = theta_det,
    mc_samples = 500L, mc_seed = 20261028L
  )$posterior
  training_threshold <- eval_with_best_threshold(
    y_train, train_predictions
  )$result$threshold[1L]
  heldout <- eval_with_best_threshold(
    y_test, fit$posterior, threshold = training_threshold
  )
  stopifnot(is.finite(heldout$result$AUC),
            heldout$result$threshold == training_threshold)
}

cat("FIBER & FIBER-Stream IRT-marginal synthetic checks passed.\n")
# Legacy Bernoulli helper no longer produces NaN with exact 0/1 rates.
legacy <- detector_likelihood(c(0, 1, 0.5, 0.5), d_test)
stopifnot(all(is.finite(legacy)), all(legacy >= 0))
