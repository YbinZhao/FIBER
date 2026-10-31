# FIBER: Fast-DetectGPT with IRT-informed Bayesian Evidence Recalibration

Research code for **FIBER** and **FIBER-Stream**, methods for detecting LLM-generated text using the continuous Fast-DetectGPT score, binary predictions from auxiliary detectors, and historical detector-response information represented by **item response theory (IRT)**.

This repository provides an **R research implementation**, rather than an installable R package. The current code implements the **IRT-marginalized detector-output likelihood** formulation. It is not the earlier version that used an IRT-informed class prior and an independent Bernoulli detector likelihood.

## Methods at a glance

- **FIBER** learns the historical class prior, class-conditional IRT item-parameter distributions, and class-conditional Fast-DetectGPT score distributions from labeled historical data. It combines these components to predict incoming texts.
- **FIBER-Stream** retains the historical labels, detector outputs, IRT item representations, and detector ability parameters, but **does not require historical Fast-DetectGPT scores**. Instead, it estimates the missing class-conditional score distributions from an incoming **unlabeled batch** using expectation-maximization (EM).
- **FIBER-Fast** and **FIBER-Stream-Fast** use the same respective procedures with **10 instead of 13 auxiliary detectors**. The research scripts remove DetectGPT, DNA-GPT, and NPR from the auxiliary set and use IRT fits for the reduced configuration.

In the core implementation, the historical AI-class proportion supplies the class prior. IRT enters the *sample-specific detector-output evidence* through marginalization, not through a separate IRT-derived prior. FIBER-Stream is a **batchwise** adaptation method, not a per-observation recursive online algorithm.

## Repository contents

| File | Purpose |
| --- | --- |
| [`Fiber_documented.R`](Fiber_documented.R) | Main `fiber()` and `fiber_stream()` functions; IRT-marginal likelihood, EM, evaluation and ROC utilities. |
| [`Fiber_test.R`](Fiber_test.R) | Self-contained synthetic checks for the current FIBER and FIBER-Stream implementations. |
| [`Fiber_multidomain.R`](Fiber_multidomain.R) | Research driver for the pooled 32,000-text, 5-fold experiment and full/reduced detector configurations. |
| [`Fiber_Domain.R`](Fiber_Domain.R) | Research driver for domain-specific 5-fold experiments, metrics and ROC figures. |
| [`README.md`](README.md) | Usage and experiment documentation. |

The repository does **not** include the underlying benchmark CSV files, the pre-fitted IRT JSON files, an IRT fitting script, or a standalone OOD evaluation driver. Obtain or prepare those inputs separately before running the experiment drivers.

## Requirements

A working R installation is required. For the **core functions and synthetic test**, install:

```r
install.packages(c("mvtnorm", "pROC"))
```

The experiment drivers additionally use these packages:

```r
install.packages(c("jsonlite", "data.table", "knitr"))
```

- `mvtnorm`: multivariate Gaussian sampling for IRT marginalization.
- `pROC`: ROC curves, AUC, and Youden threshold selection.
- `jsonlite`: reading fitted IRT parameters from JSON.
- `data.table`, `knitr`: loaded by the multi-domain research script.

The code uses base R for the remaining numerical and plotting operations.

## Quick start: run the included synthetic checks

Clone the repository and run the supplied test script:

```bash
git clone https://github.com/YbinZhao/FIBER.git
cd FIBER
Rscript Fiber_test.R
```

Alternatively, in an R session:

```r
source("Fiber_test.R")
```

`Fiber_test.R` constructs synthetic labels, detector outputs, IRT item representations, detector abilities, and Fast-DetectGPT-like scores. It checks posterior ranges, shared IRT likelihoods, reproducible Monte Carlo evaluation, and EM outputs. **It does not estimate IRT parameters from real data and does not reproduce the reported experimental results.**

## Data and fitted IRT parameters

The core functions require precomputed detector predictions and fitted historical IRT parameters. In particular, **both** `fiber()` and `fiber_stream()` require `theta_det`; omitting it produces an error in the current implementation.

| Object | Required dimensions | Meaning |
| --- | --- | --- |
| `y_train` | `n_train` | Historical labels: `0` = human, `1` = AI-generated. |
| `d_train` / `D_train` | `n_train × K` | Historical binary auxiliary-detector predictions. |
| `z_train` / `Z_train` | `n_train × (2*d)` | Fitted historical IRT item parameters, ordered as all `alpha` coordinates followed by all `beta` coordinates. |
| `theta_det` | `K × d` | Fitted IRT detector-ability matrix, with detector rows aligned to the prediction columns. |
| `s_train` | `n_train` | Historical continuous Fast-DetectGPT scores (**FIBER only**). |
| `s_test` / `s_stream` | `n_new` | Incoming continuous Fast-DetectGPT scores. |
| `d_test` / `D_stream` | `n_new × K` | Incoming binary auxiliary-detector predictions. |

For historical text `i` and detector `k`, the correctness response used to fit IRT is

`R[i, k] = as.integer(D_train[i, k] == y_train[i])`.

**Before calling the FIBER functions, fit the multidimensional 2PL IRT model separately on this historical correctness matrix.** The fitted item representations and detector abilities must come from the **same IRT fit**. The new texts' labels and IRT item parameters are not needed for prediction.

The provided experiment scripts read IRT JSON objects with fields `disc`, `diff`, and `ability`, for example:

```r
library(jsonlite)
irt <- jsonlite::fromJSON("path/to/irt_13_d5/best_parameters.json")
z_train <- cbind(as.matrix(irt$disc), as.matrix(irt$diff))
theta_det <- as.matrix(irt$ability)
```

This example assumes the JSON item rows are aligned with **your selected historical training rows**. Verify row identity, detector ordering, and the expected dimensions; matrix sizes alone cannot establish correct alignment.

For the paper's standard configuration, `K = 13`; the reduced configuration uses `K = 10`. The experiment scripts use `d = 5`. **Do not include the thresholded Fast-DetectGPT prediction among the auxiliary detector columns**: its continuous score is already a separate evidence source.

## Using the FIBER functions

Load the implementation:

```r
source("Fiber_documented.R")
```

The examples below assume that the objects listed in the preceding table have been prepared and aligned.

### Standard FIBER

```r
fit <- fiber(
  s_train = s_train,
  y_train = y_train,
  d_train = d_train,
  z_train = z_train,
  s_test = s_test,
  d_test = d_test,
  theta_det = theta_det,      # Required: K x d from the historical IRT fit
  mc_samples = 2000L,        # Monte Carlo samples per class
  mc_seed = 20261028L,
  threshold = 0.5
)

head(fit$posterior)            # Estimated AI-class probabilities
head(fit$y_pred)               # Predicted labels at the supplied threshold
fit$pi_prior                  # Historical AI-class proportion
```

Standard FIBER fits the two Gaussian score densities from historical `s_train` and integrates the detector-output likelihood over the class-conditional fitted IRT item distributions. The default evidence exponents are `w = c(1, 1, 1)` for the prior log-odds, score log-likelihood ratio, and IRT detector-output log-likelihood ratio. A non-default `w` changes the evidence weighting relative to the default Bayesian formulation.

### FIBER-Stream

```r
stream_fit <- fiber_stream(
  y_train = y_train,
  D_train = d_train,
  Z_train = z_train,
  s_stream = s_test,
  D_stream = d_test,
  theta_det = theta_det,      # Required: same historical IRT fit
  mc_samples = 2000L,
  mc_seed = 20261028L,
  max_iter = 500L,
  tol = 1e-6,
  threshold = 0.5
)

head(stream_fit$posterior)
head(stream_fit$y_pred)
stream_fit$em$converged
stream_fit$em$iterations
stream_fit$em$trace            # EM log-likelihood and parameter trace
stream_fit$parameters$score   # Fitted class-conditional score parameters
```

`fiber_stream()` holds the historical class prior and IRT detector evidence fixed while updating the **two Gaussian score distributions** using the current unlabeled batch. The EM implementation enforces the expected AI-score mean to be no smaller than the human-score mean; its `sigma_as` and `sigma_hs` fields are **variances**, not standard deviations. EM may terminate at the maximum iteration count without convergence; check the returned diagnostics.

### Returned objects and compatibility notes

Both main functions return `posterior`, `y_pred`, `pi_prior`, `parameters`, and `likelihoods`. FIBER-Stream additionally returns `em` and `omega_init`.

- The current prior is `pi_prior = mean(y_train)`. The returned `pi_irt` is only a **deprecated compatibility alias** for that same empirical proportion; it is **not** an IRT-informed prior. `eta` is `NULL`.
- Deprecated arguments (`a_0`, `b_0`, and, for FIBER-Stream, `pi_0`) do not define the current prior.
- Prediction uses the strict comparison `posterior > threshold`.
- `mc_seed` makes the IRT Monte Carlo integration reproducible while preserving the caller's prior R random-number generator state.
- `mc_samples`, `batch_size`, `cov_ridge`, and `score_variance_floor` control numerical approximation and stability.

## Evaluation, thresholds and ROC curves

The implementation provides:

- `eval_with_best_threshold(y_true, p_hat, method = "youden", threshold = NULL)` for AUC and accuracy, precision, recall, specificity, F1, and confusion-matrix counts.
- `plot_cv_roc_three_methods(...)` for a **three-method** comparison of FIBER, FIBER-Stream, and Fast-DetectGPT. Its posterior inputs are per-fold matrices ordered by the held-out rows of each fold; Fast-DetectGPT is supplied as a full-length score vector.

If `threshold = NULL`, `eval_with_best_threshold()` selects a Youden-optimal cutoff **using the labels supplied to that call**. For ordinary held-out evaluation, select the cutoff on labeled training/validation predictions and pass it unchanged to the test evaluation:

```r
train_fit <- fiber(
  s_train = s_train, y_train = y_train,
  d_train = d_train, z_train = z_train,
  s_test = s_train, d_test = d_train,
  theta_det = theta_det
)
cutoff <- eval_with_best_threshold(y_train, train_fit$posterior)$result$threshold[1]

heldout_metrics <- eval_with_best_threshold(
  y_true = y_test,
  p_hat = fit$posterior,
  threshold = cutoff
)$result
print(heldout_metrics)
```

Here `y_test` is used **only to compute evaluation metrics**, not to select the cutoff. The documented paper's **OOD evaluation** instead uses *retrospective Youden threshold selection on the labeled OOD test set* for methods with continuous scores, to match its baseline comparison protocol. Those threshold-dependent OOD values are **post-hoc calibrated results**, not a label-free deployment estimate. AUC does not depend on the chosen cutoff. **No standalone OOD evaluation script is currently included in this repository.**

## Research experiment scripts

### Multi-domain experiment: `Fiber_multidomain.R`

This script implements the research workflow for a pooled benchmark of **32,000 texts**, with **five folds** and 13-detector / 10-detector configurations. It expects external files such as:

```text
all_detector_scores_32000_14.csv
cv_5fold_d5/
  fold_1/
    D_32000_13.csv
    test_sample_ids.csv
    irt_13_d5/best_parameters.json
    irt_10_d5/best_parameters.json
  ... fold_2 through fold_5 ...
```

The 13-detector matrix excludes the Fast-DetectGPT binary decision. In the reduced configuration the script removes columns `2, 3, 8` (corresponding to the named detectors only **for its original column order**) and uses the separately fitted 10-detector IRT parameters. It writes train/test posterior CSV files, including names such as `fiber_posterior_test.csv`, `fiber_stream_posterior_test.csv`, and their `_short_` counterparts.

**Before running:** update the machine-specific `setwd()` path, place the required external datasets and fold-level IRT fits in the expected locations, and verify row indices and any CSV row-index columns. This is a research driver, not a turnkey data downloader or IRT fitting pipeline.

### Domain-specific experiment: `Fiber_Domain.R`

This driver evaluates arXiv, Writing Prompts, XSum, and Yelp separately using **8,000 texts per domain**, **5 folds**, and **6,400/1,600 training/test texts per fold**. It expects domain-specific training-fold IRT fits (do not substitute a fit trained on a pooled multi-domain fold) and per-domain fold IDs. Its configuration includes:

```r
project_dir <- "/path/to/your/FIBER/data"
domains_to_run <- c("arxiv", "writing_prompt", "xsum", "yelp_reviews")
# Update fold_id_files and fold_dir() to match your files.
```

The committed script currently sets `domains_to_run <- c("arxiv")` by default. Expected paths include:

```text
cv_domain_5fold_d5/arxiv/fold_1/D_8000_13.csv
cv_domain_5fold_d5/arxiv/fold_1/irt_13_d5/best_parameters.json
cv_domain_5fold_d5/arxiv/fold_1/irt_10_d5/best_parameters.json
```

Training-fold Youden cutoffs are applied unchanged to held-out test predictions. When inputs are present, the script writes per-domain metrics, prediction CSVs, EM diagnostics, training thresholds, ROC PDFs/PNGs, and aggregate files under `FIBER_domain_results/`. It creates the four-domain composite ROC PDF only when all four domains have been processed.

## Practical checks and limitations

- **Input alignment matters:** `y_train`, score vectors, detector matrices, `Z_train` and `theta_det` must have consistent sample/detector order. A successful dimension check is not a substitute for matching IDs.
- **No test-label-dependent IRT fitting:** estimate the IRT parameters only from the appropriate historical training data/fold.
- **Approximation assumptions:** the current framework uses Gaussian class-conditional IRT representations and Gaussian Fast-DetectGPT scores; detector outputs are independent conditional on the latent IRT representation and class, and the score and detector-output profile are modeled as conditionally independent given the class.
- **Parameter uncertainty:** fitted historical parameters are used as plug-in estimates.
- **Streaming scope:** FIBER-Stream adapts a batch of unlabeled scores with EM rather than updating model parameters recursively per observation.
- **Reproducibility:** the provided repository does not bundle the original training/benchmark data, historical IRT-fit generation procedure, or OOD script. Running the synthetic smoke test validates program behavior but does not independently reproduce the paper's reported tables.

## Data sources and attribution

The associated study uses historical detector-evaluation data and external Fast-DetectGPT benchmarks. Consult the source repositories for downloads and their respective usage conditions:

1. [text-detector-evaluation](https://github.com/YbinZhao/text-detector-evaluation) — multi-domain detector evaluation and IRT benchmarking resources.
2. [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) — the method and datasets used for the external evaluation.

## References and citation

- Zhao, Y., Chen, Z., Bao, G., Zhang, G., and Li, Y. (2026). *A measurement perspective on benchmarking LLM-generated text detectors using item response theory.* arXiv:2610.24972. [Paper](https://arxiv.org/abs/2610.24972) | [Repository](https://github.com/YbinZhao/text-detector-evaluation).
- Bao, G., Zhao, Y., Teng, Z., Yang, L., and Zhang, Y. (2024). *Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature.* ICLR 2024. [Paper](https://openreview.net/forum?id=Bpcgcr8E8Z) | [Repository](https://github.com/baoguangsheng/fast-detect-gpt).

If you use **FIBER** itself, please also cite *FIBER: Fast-DetectGPT with IRT-informed Bayesian Evidence Recalibration* (manuscript; add the final publication/DOI when available).
