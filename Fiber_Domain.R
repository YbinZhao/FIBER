# =============================================================================
# FIBER: domain-specific 5-fold evaluation + ROC curves
# =============================================================================
# Dataset: all_detector_scores_32000_14.csv (32,000 rows; 8,000 per domain)
# Domains: arxiv, writing_prompt, xsum, yelp_reviews
# Each domain: 5 folds x 1,600 test items, 6,400 training items.
# Model variants: FIBER, FIBER-Fast, FIBER-Stream, FIBER-Stream-Fast.
#
# IMPORTANT: The IRT parameters MUST have been fitted independently for each
# DOMAIN and TRAINING FOLD (6,400 texts). Do not reuse 25,600-row IRT parameters
# fitted on the pooled 32,000-text experiment.
#
# REQUIRED domain-specific directory layout (edit fold_dir() if different):
#   cv_domain_5fold_d5/arxiv/fold_1/D_8000_13.csv
#   cv_domain_5fold_d5/arxiv/fold_1/irt_13_d5/best_parameters.json
#   cv_domain_5fold_d5/arxiv/fold_1/irt_10_d5/best_parameters.json
#   ... five folds for each domain ...
#
# REQUIRED domain fold CSV, with 8,000 rows and fold IDs 1:5:
#   fold_id in 5 fold_arxiv.csv            [provided by the user]
#   fold_id in 5 fold_writing_prompt.csv  [change path if differently named]
#   fold_id in 5 fold_xsum.csv            [change path if differently named]
#   fold_id in 5 fold_yelp_reviews.csv    [change path if differently named]
#
# The arxiv file has 2 columns: an R-export row index and `x` (fold IDs).
# The first column is NOT the fold number; use column `x`.
#
# Threshold protocol: choose Youden on fitted TRAINING predictions, then apply
# that fixed threshold on HELD-OUT TEST predictions (no test-label tuning).
# FIBER-Stream fits a separate unlabeled-batch EM on each test fold.
# =============================================================================

# -------- 1. EDIT THIS CONFIGURATION FOR YOUR COMPUTER ---------------------
project_dir <- "/Users/yan/Desktop/同步文件夹/FIBER_update/FIBER_IRT"
setwd(project_dir)

source("Fiber_documented.R")   # Must include the fixed IRT matrix-dimension code
if (!requireNamespace("pROC", quietly = TRUE)) stop("Install pROC")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite")

# Run arxiv first; after adding the other domains' fold CSVs and IRT files,
# replace with: c("arxiv", "writing_prompt", "xsum", "yelp_reviews")
domains_to_run <- c("arxiv")
all_domains <- c("arxiv", "writing_prompt", "xsum", "yelp_reviews")

fold_id_files <- c(
  arxiv = "fold_id in 5 fold_arxiv.csv",
  writing_prompt = "fold_id in 5 fold_writing_prompt.csv",
  xsum = "fold_id in 5 fold_xsum.csv",
  yelp_reviews = "fold_id in 5 fold_yelp_reviews.csv"
)

# CHANGE ONLY THIS PATH FUNCTION if your domain-specific IRT folders are elsewhere.
fold_dir <- function(domain, fold) {
  file.path("cv_domain_5fold_d5", domain, paste0("fold_", fold))
}

# Set TRUE only if you deliberately want to subset a pooled 32,000-row
# detector-output file to the selected domain. Beware: pooled thresholding may
# have used information outside the domain-specific training fold.
allow_pooled_detector_matrix <- FALSE

N_folds <- 5L
mc_samples <- 2000L
mc_seed <- 20261028L
output_root <- "FIBER_domain_results"
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

methods <- c("FIBER", "FIBER-Fast", "FIBER-Stream", "FIBER-Stream-Fast")
method_colors <- c("red", "#9B111E", "#E69F00", "#FF7F0E")
score_color <- "blue"
fpr_grid <- seq(0, 1, by = 0.005)

# -------- 2. INPUT AND VALIDATION HELPERS ---------------------------------
read_domain_fold_ids <- function(path, n = 8000L, K = 5L) {
  if (!file.exists(path)) stop("Missing domain fold-ID file: ", path)
  tbl <- read.csv(path, check.names = FALSE)
  if (nrow(tbl) != n) stop("Fold file ", path, " must have ", n, " rows.")
  if ("x" %in% names(tbl)) {
    folds <- tbl[["x"]]
  } else if (ncol(tbl) == 1L) {
    folds <- tbl[[1L]]
  } else {
    stop("Expected a fold column named 'x', or a single fold column: ", path)
  }
  if (!is.numeric(folds) || anyNA(folds) ||
      any(!folds %in% seq_len(K))) {
    stop("Fold IDs must be integers in 1,...,", K, ": ", path)
  }
  folds <- as.integer(folds)
  if (any(tabulate(folds, nbins = K) != n/K)) {
    stop("This analysis expects equal fold sizes (1,600 per fold).")
  }
  folds
}

read_domain_detector_matrix <- function(domain, f, global_rows, n_domain) {
  folder <- fold_dir(domain, f)
  path_domain <- file.path(folder, sprintf("D_%d_13.csv", n_domain))
  path_pooled <- file.path(folder, "D_32000_13.csv")
  if (file.exists(path_domain)) {
    path <- path_domain
    use_pooled <- FALSE
  } else if (allow_pooled_detector_matrix && file.exists(path_pooled)) {
    path <- path_pooled
    use_pooled <- TRUE
  } else {
    stop("Fold-specific detector matrix missing: ", path_domain,
         "\nSet fold_dir() to your actual domain-specific output folder.")
  }
  tbl <- read.csv(path, check.names = FALSE)
  # Drop an automatically written row index, if present.
  if (ncol(tbl) == 14L) {
    first <- suppressWarnings(as.numeric(tbl[[1L]]))
    if (!anyNA(first) &&
        (identical(first, as.numeric(seq_len(nrow(tbl)))) ||
         identical(first, as.numeric(seq.int(0L, nrow(tbl)-1L))))) {
      tbl <- tbl[-1L]
    }
  }
  if (ncol(tbl) != 13L) stop("Expected 13 detector columns in: ", path)
  if (use_pooled) {
    if (nrow(tbl) != 32000L) stop("Pooled D matrix must have 32,000 rows: ", path)
    tbl <- tbl[global_rows, , drop = FALSE]
  }
  if (nrow(tbl) != n_domain) stop("D matrix has wrong row count: ", path)
  D <- as.matrix(tbl)
  storage.mode(D) <- "double"
  if (any(!is.finite(D)) || any(!(D %in% c(0, 1)))) {
    stop("D must be a finite binary (0/1) matrix: ", path)
  }
  D
}

read_domain_irt <- function(domain, f, K_det, train_ids, D) {
  path <- file.path(fold_dir(domain, f),
                    sprintf("irt_%d_d5", K_det), "best_parameters.json")
  if (!file.exists(path)) stop("Missing DOMAIN-SPECIFIC IRT fit: ", path)
  model <- jsonlite::fromJSON(path)
  if (is.null(model$disc) || is.null(model$diff) || is.null(model$ability)) {
    stop("JSON must contain disc, diff and ability: ", path)
  }
  Z <- cbind(as.matrix(model$disc), as.matrix(model$diff))
  theta <- as.matrix(model$ability)
  storage.mode(Z) <- "double"
  storage.mode(theta) <- "double"
  if (nrow(Z) != length(train_ids) || nrow(theta) != K_det ||
      ncol(Z) != 2L * ncol(theta) || ncol(theta) != 5L ||
      any(!is.finite(Z)) || any(!is.finite(theta))) {
    stop("IRT parameter sizes inconsistent for ", domain, " fold ", f,
         " (expected ", length(train_ids), " x 10 items and ", K_det,
         " x 5 abilities): ", path)
  }
  if (!is.null(colnames(D)) && !is.null(rownames(theta)) &&
      !identical(colnames(D), rownames(theta))) {
    stop("IRT detector ability order does not match D column order: ", path)
  }
  # Critical: Z row i must correspond to domain train_ids[i] (in ascending
  # domain_index order). Dimensions cannot prove alignment: confirm externally.
  list(Z = Z, theta = theta)
}

get_youden_threshold <- function(y, p) {
  roc_obj <- pROC::roc(y, p, levels = c(0, 1),
                       direction = "<", quiet = TRUE)
  cands <- pROC::coords(roc_obj, x = "best", best.method = "youden",
                        ret = "threshold", transpose = FALSE)
  cutoffs <- as.numeric(unlist(cands, use.names = FALSE))
  cutoffs <- cutoffs[is.finite(cutoffs)]
  if (!length(cutoffs)) stop("No finite Youden threshold available")
  cutoffs[[1L]]
}

score_metrics <- function(y, p, cutoff, method, domain, f) {
  if (length(y) != length(p) || anyNA(y) || any(!is.finite(p)) ||
      any(!y %in% c(0, 1))) stop("Invalid prediction/label alignment")
  # FIBER uses > threshold; align evaluation with fiber() and fiber_stream().
  pred <- as.integer(p > cutoff)
  TP <- sum(pred == 1 & y == 1)
  FP <- sum(pred == 1 & y == 0)
  FN <- sum(pred == 0 & y == 1)
  TN <- sum(pred == 0 & y == 0)
  safe_ratio <- function(a, b) if (b == 0) NA_real_ else a / b
  acc <- safe_ratio(TP + TN, length(y))
  prec <- safe_ratio(TP, TP + FP)
  rec <- safe_ratio(TP, TP + FN)
  f1 <- safe_ratio(2 * TP, 2 * TP + FP + FN)
  auc <- as.numeric(pROC::auc(pROC::roc(y, p,
                 levels = c(0, 1), direction = "<", quiet = TRUE)))
  data.frame(Domain = domain, Fold = f, Method = method,
             N = length(y), N_AI = sum(y == 1), N_Human = sum(y == 0),
             Threshold = cutoff, TP = TP, FP = FP, FN = FN, TN = TN,
             Accuracy = acc, Precision = prec, Recall = rec,
             Specificity = safe_ratio(TN, TN + FP), F1 = f1, AUC = auc)
}

# -------- 3. ROC PLOTTING -------------------------------------------------
get_fold_roc_data <- function(y, p) {
  r <- pROC::roc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE)
  tpr <- as.numeric(pROC::coords(
    r, x = 1 - fpr_grid, input = "specificity",
    ret = "sensitivity", transpose = FALSE
  )$sensitivity)
  list(auc = as.numeric(pROC::auc(r)), tpr = tpr)
}

plot_domain_roc_panel <- function(roc_data, domain, show_legend = TRUE) {
  # Five faint dashed lines + one mean solid line for each of 4 FIBER methods,
  # plus the Fast-DetectGPT baseline.
  plot(NA_real_, NA_real_, xlim = c(0, 1), ylim = c(0, 1),
       xaxs = "i", yaxs = "i", xlab = "False Positive Rate",
       ylab = "True Positive Rate", main = domain)
  abline(0, 1, col = "gray70", lty = 3)
  all_names <- c(methods, "Fast-DetectGPT")
  labels <- character(length(all_names))
  all_cols <- c(method_colors, score_color)
  for (i in seq_along(all_names)) {
    name <- all_names[[i]]
    x <- roc_data[[name]]
    for (f in seq_len(ncol(x$tpr))) {
      lines(fpr_grid, x$tpr[, f], col = grDevices::adjustcolor(
        all_cols[[i]], alpha.f = 0.23), lty = 2, lwd = 1)
    }
    lines(fpr_grid, rowMeans(x$tpr), col = all_cols[[i]], lwd = 2.4)
    labels[[i]] <- sprintf("%s: %.3f (%.3f)", name,
                           mean(x$auc), stats::sd(x$auc))
  }
  if (show_legend) {
    legend("bottomright", legend = labels, col = all_cols,
           lwd = 2, cex = 0.7, bty = "n")
  }
}

# -------- 4. LOAD COMPLETE DATA -------------------------------------------
score_file <- "all_detector_scores_32000_14.csv"
if (!file.exists(score_file)) stop("Cannot find: ", score_file)
scores <- read.csv(score_file, check.names = FALSE)
required_fields <- c("sample_id", "domain", "domain_index", "label",
                     "fast_detect_gpt")
if (!all(required_fields %in% names(scores))) {
  stop("Score CSV missing columns: ",
       paste(setdiff(required_fields, names(scores)), collapse = ", "))
}
if (nrow(scores) != 32000L || anyDuplicated(scores$sample_id) ||
    any(!is.finite(scores$fast_detect_gpt)) ||
    any(!scores$label %in% c(0, 1))) {
  stop("Score CSV invalid; check N, sample IDs, labels and scores")
}
if (!all(domains_to_run %in% all_domains)) stop("Unknown domain selection")

all_fold_metrics <- list()
all_predictions <- list()
all_EM <- list()
all_thresholds <- list()
all_domain_rocs <- list()

# -------- 5. DOMAIN x FOLD x MODEL EXPERIMENT -----------------------------
for (domain in domains_to_run) {
  cat("\n========== DOMAIN:", domain, "==========\n")
  global_rows <- which(scores$domain == domain)
  domain_df <- scores[global_rows, , drop = FALSE]
  ord <- order(domain_df$domain_index)
  global_rows <- global_rows[ord]
  domain_df <- domain_df[ord, , drop = FALSE]
  n_domain <- nrow(domain_df)
  if (n_domain != 8000L ||
      !identical(as.integer(domain_df$domain_index), 0:7999)) {
    stop("Expected 8,000 texts with domain_index 0:7999: ", domain)
  }
  y_domain <- as.integer(domain_df$label)
  s_domain <- as.numeric(domain_df$fast_detect_gpt)
  folds <- read_domain_fold_ids(fold_id_files[[domain]], n_domain, N_folds)
  print(table(Fold = folds, Label = y_domain))

  dir_domain <- file.path(output_root, domain)
  dir.create(dir_domain, showWarnings = FALSE, recursive = TRUE)
  n_test <- 1600L
  n_train <- 6400L
  train_prob <- setNames(lapply(methods, function(v) matrix(NA_real_, n_train, N_folds)), methods)
  test_prob <- setNames(lapply(methods, function(v) matrix(NA_real_, n_test, N_folds)), methods)
  thresholds <- matrix(NA_real_, nrow = N_folds, ncol = length(methods),
                       dimnames = list(paste0("fold_", 1:N_folds), methods))
  roc_data <- setNames(lapply(c(methods, "Fast-DetectGPT"), function(x) {
    list(tpr = matrix(NA_real_, length(fpr_grid), N_folds),
         auc = rep(NA_real_, N_folds))
  }), c(methods, "Fast-DetectGPT"))
  fold_metrics <- list()
  predictions <- list()
  em_diag <- list()

  for (f in seq_len(N_folds)) {
    cat("\n-- ", domain, " fold ", f, "/", N_folds, " --\n", sep = "")
    test_ids <- which(folds == f)     # domain-local, 1-based, NO +1
    train_ids <- which(folds != f)    # sorted, for alignment with IRT Z
    if (length(test_ids) != n_test || length(train_ids) != n_train ||
        length(unique(y_domain[test_ids])) != 2L) {
      stop("Invalid train/test fold sizes or only one test class")
    }
    y_train <- y_domain[train_ids]
    y_test <- y_domain[test_ids]
    s_train <- s_domain[train_ids]
    s_test <- s_domain[test_ids]
    D_full <- read_domain_detector_matrix(domain, f, global_rows, n_domain)

    for (variant in c("full", "fast")) {
      D <- if (variant == "full") D_full else D_full[, -c(2L, 3L, 8L), drop = FALSE]
      irt <- read_domain_irt(domain, f, ncol(D), train_ids, D)
      D_train <- D[train_ids, , drop = FALSE]
      D_test <- D[test_ids, , drop = FALSE]
      method_standard <- if (variant == "full") "FIBER" else "FIBER-Fast"
      method_stream <- if (variant == "full") "FIBER-Stream" else "FIBER-Stream-Fast"
      seed <- mc_seed + 1000L*f + if (variant == "full") 0L else 100L

      # PHASE A: fit on domain TRAINING items and choose Youden threshold.
      fit_train <- fiber(
        s_train = s_train, y_train = y_train,
        d_train = D_train, z_train = irt$Z,
        s_test = s_train, d_test = D_train,
        theta_det = irt$theta,
        mc_samples = mc_samples, mc_seed = seed
      )
      stream_train <- fiber_stream(
        y_train = y_train, D_train = D_train, Z_train = irt$Z,
        s_stream = s_train, D_stream = D_train,
        theta_det = irt$theta,
        mc_samples = mc_samples, mc_seed = seed
      )
      cutoff_std <- get_youden_threshold(y_train, fit_train$posterior)
      cutoff_stream <- get_youden_threshold(y_train, stream_train$posterior)
      thresholds[f, method_standard] <- cutoff_std
      thresholds[f, method_stream] <- cutoff_stream
      train_prob[[method_standard]][, f] <- fit_train$posterior
      train_prob[[method_stream]][, f] <- stream_train$posterior

      # PHASE B: held-out TEST predictions; thresholds remain unchanged.
      fit_test <- fiber(
        s_train = s_train, y_train = y_train,
        d_train = D_train, z_train = irt$Z,
        s_test = s_test, d_test = D_test,
        threshold = cutoff_std, theta_det = irt$theta,
        mc_samples = mc_samples, mc_seed = seed
      )
      stream_test <- fiber_stream(
        y_train = y_train, D_train = D_train, Z_train = irt$Z,
        s_stream = s_test, D_stream = D_test,
        threshold = cutoff_stream, theta_det = irt$theta,
        mc_samples = mc_samples, mc_seed = seed
      )

      for (method in c(method_standard, method_stream)) {
        obj <- if (method == method_standard) fit_test else stream_test
        cutoff <- thresholds[f, method]
        if (length(obj$posterior) != length(test_ids) ||
            any(!is.finite(obj$posterior)) ||
            any(!isTRUE(all.equal(obj$y_pred, as.numeric(obj$posterior > cutoff))))) {
          stop("Invalid posterior or prediction: ", domain, ", ", method, ", fold ", f)
        }
        test_prob[[method]][, f] <- obj$posterior
        key <- paste(f, method, sep = "_")
        fold_metrics[[key]] <- score_metrics(y_test, obj$posterior,
                                              cutoff, method, domain, f)
        predictions[[key]] <- data.frame(
          Domain = domain, Fold = f, Method = method,
          Domain_Index_0based = domain_df$domain_index[test_ids],
          Global_Sample_ID_0based = domain_df$sample_id[test_ids],
          True_Label = y_test, Fast_Score = s_test,
          Posterior = obj$posterior, Predicted_Label = obj$y_pred,
          Train_Threshold = cutoff
        )
        r <- get_fold_roc_data(y_test, obj$posterior)
        roc_data[[method]]$tpr[, f] <- r$tpr
        roc_data[[method]]$auc[f] <- r$auc
      }
      for (phase in c("train", "test")) {
        em <- if (phase == "train") stream_train$em else stream_test$em
        key <- paste(f, method_stream, phase, sep = "_")
        em_diag[[key]] <- data.frame(
          Domain = domain, Fold = f, Method = method_stream,
          Phase = phase, Converged = em$converged,
          Iterations = em$iterations, Mu_AI = em$mu_as,
          Mu_Human = em$mu_hs, Var_AI = em$sigma_as,
          Var_Human = em$sigma_hs, LogLik = em$loglik
        )
      }
    }

    # Common Fast-DetectGPT baseline for the two detector configurations.
    r0 <- get_fold_roc_data(y_test, s_test)
    roc_data[["Fast-DetectGPT"]]$tpr[, f] <- r0$tpr
    roc_data[["Fast-DetectGPT"]]$auc[f] <- r0$auc
  }

  metrics_df <- do.call(rbind, fold_metrics)
  rownames(metrics_df) <- NULL
  preds_df <- do.call(rbind, predictions)
  rownames(preds_df) <- NULL
  em_df <- do.call(rbind, em_diag)
  rownames(em_df) <- NULL
  summary_df <- do.call(rbind, lapply(methods, function(name) {
    x <- metrics_df[metrics_df$Method == name, , drop = FALSE]
    vars <- c("Accuracy", "Precision", "Recall", "F1", "AUC")
    out <- data.frame(Domain = domain, Method = name)
    for (v in vars) {
      out[[paste0(v, "_mean")]] <- mean(x[[v]], na.rm = TRUE)
      out[[paste0(v, "_sd")]] <- sd(x[[v]], na.rm = TRUE)
    }
    out
  }))

  # Same 1,600 x 5 CSV layout used in the pooled experiment.
  prefix <- c("FIBER" = "fiber_posterior",
              "FIBER-Stream" = "fiber_stream_posterior",
              "FIBER-Fast" = "fiber_posterior_short",
              "FIBER-Stream-Fast" = "fiber_stream_posterior_short")
  for (method in methods) {
    write.csv(train_prob[[method]], file.path(dir_domain,
              paste0(prefix[[method]], "_train.csv")))
    write.csv(test_prob[[method]], file.path(dir_domain,
              paste0(prefix[[method]], "_test.csv")))
  }
  write.csv(metrics_df, file.path(dir_domain, "fold_metrics.csv"), row.names = FALSE)
  write.csv(summary_df, file.path(dir_domain, "summary_mean_sd.csv"), row.names = FALSE)
  write.csv(preds_df, file.path(dir_domain, "heldout_predictions.csv"), row.names = FALSE)
  write.csv(as.data.frame(thresholds), file.path(dir_domain, "training_thresholds.csv"))
  write.csv(em_df, file.path(dir_domain, "stream_em_diagnostics.csv"), row.names = FALSE)
  write.csv(data.frame(Fold = 1:N_folds,
                       AUC = roc_data[["Fast-DetectGPT"]]$auc),
            file.path(dir_domain, "fast_detect_gpt_fold_auc.csv"), row.names = FALSE)
  saveRDS(list(domain = domain, fold_id = folds, test_prob = test_prob,
               threshold = thresholds, roc = roc_data, summary = summary_df),
          file.path(dir_domain, "domain_results.rds"))

  grDevices::pdf(file.path(dir_domain, "domain_5fold_ROC.pdf"),
                 width = 8, height = 6)
  plot_domain_roc_panel(roc_data, domain)
  grDevices::dev.off()
  grDevices::png(file.path(dir_domain, "domain_5fold_ROC.png"),
                 width = 2400, height = 1800, res = 300)
  plot_domain_roc_panel(roc_data, domain)
  grDevices::dev.off()

  all_fold_metrics[[domain]] <- metrics_df
  all_predictions[[domain]] <- preds_df
  all_EM[[domain]] <- em_df
  all_thresholds[[domain]] <- data.frame(Domain = domain,
      Fold = seq_len(N_folds), thresholds, check.names = FALSE)
  all_domain_rocs[[domain]] <- roc_data
  cat("\nHeld-out five-fold metrics for ", domain, ":\n", sep = "")
  print(summary_df, row.names = FALSE)
}

# -------- 6. COMBINED DOMAIN RESULTS --------------------------------------
if (length(all_fold_metrics)) {
  metrics_all <- do.call(rbind, all_fold_metrics)
  em_all <- do.call(rbind, all_EM)
  predictions_all <- do.call(rbind, all_predictions)
  thresholds_all <- do.call(rbind, all_thresholds)
  write.csv(metrics_all, file.path(output_root, "ALL_domain_fold_metrics.csv"),
            row.names = FALSE)
  write.csv(predictions_all,
            file.path(output_root, "ALL_domain_heldout_predictions.csv"),
            row.names = FALSE)
  write.csv(em_all, file.path(output_root, "ALL_domain_EM_diagnostics.csv"),
            row.names = FALSE)
  write.csv(thresholds_all,
            file.path(output_root, "ALL_domain_training_thresholds.csv"),
            row.names = FALSE)
  summary_all <- do.call(rbind, lapply(split(metrics_all,
                          interaction(metrics_all$Domain, metrics_all$Method,
                                      drop = TRUE)), function(x) {
    out <- data.frame(Domain = x$Domain[1], Method = x$Method[1])
    for (v in c("Accuracy", "Precision", "Recall", "F1", "AUC")) {
      out[[paste0(v, "_mean")]] <- mean(x[[v]], na.rm = TRUE)
      out[[paste0(v, "_sd")]] <- stats::sd(x[[v]], na.rm = TRUE)
    }
    out
  }))
  rownames(summary_all) <- NULL
  write.csv(summary_all,
            file.path(output_root, "ALL_domain_summary_mean_sd.csv"),
            row.names = FALSE)

  if (length(all_domain_rocs) == 4L) {
    grDevices::pdf(file.path(output_root, "Figure2_four_domains_ROC.pdf"),
                   width = 14, height = 10)
    old_par <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
    for (dom in all_domains) plot_domain_roc_panel(all_domain_rocs[[dom]], dom)
    par(old_par)
    grDevices::dev.off()
  }
  cat("\nALL domain results saved under: ",
      normalizePath(output_root), "\n", sep = "")
}
