# FIBER: IRT-Informed Bayesian Evidence Recalibration

Research code for **FIBER** and **FIBER-Stream**, two methods for detecting LLM-generated text by combining the continuous Fast-DetectGPT score, auxiliary detector predictions, and historical detector-response information summarized through item response theory (IRT).

- **FIBER** uses labeled historical Fast-DetectGPT scores to estimate class-conditional score distributions.
- **FIBER-Stream** retains historical labels, detector outputs, and fitted IRT item parameters, but estimates the score distributions from an incoming unlabeled batch using expectation-maximization (EM).

This repository contains an R research implementation, not a packaged R library.

## Requirements

Install the required R packages:

```r
install.packages(c("mvtnorm", "pROC"))
```

- `mvtnorm` is used to evaluate multivariate Gaussian densities.
- `pROC` is used by the ROC/AUC and threshold-selection utilities.

## Getting started

Load the functions:

```r
source("Fiber_documented.R")
```

Prepare the following objects before running the methods:

| Input | Description |
| --- | --- |
| `y_train` | Historical labels: `0` = human-written, `1` = AI-generated. |
| `d_train` | Historical binary predictions from auxiliary detectors (rows = texts, columns = detectors). |
| `z_train` | Fitted historical IRT item-parameter matrix, aligned with `y_train`. |
| `s_train` | Historical continuous Fast-DetectGPT scores (FIBER only). |
| `s_test` | Continuous Fast-DetectGPT scores for incoming texts. |
| `d_test` | Auxiliary detector predictions for incoming texts; columns match `d_train`. |

**Important:** Fit the IRT model separately using the historical detector-correctness matrix before calling these functions. IRT fitting is not included in this script. When reproducing the FIBER configuration, do not include the thresholded Fast-DetectGPT decision among the auxiliary detectors, because the continuous Fast-DetectGPT score is already used as separate evidence.

### FIBER

```r
fit <- fiber(
  s_train = s_train,
  y_train = y_train,
  d_train = d_train,
  z_train = z_train,
  s_test = s_test,
  d_test = d_test,
  threshold = 0.5
)

head(fit$posterior)  # Estimated P(AI-generated | evidence)
head(fit$y_pred)     # Binary predictions
```

### FIBER-Stream

Unlike FIBER, FIBER-Stream does **not** require historical Fast-DetectGPT scores. Its EM estimation uses the incoming texts as an unlabeled batch.

```r
stream_fit <- fiber_stream(
  y_train = y_train,
  D_train = d_train,
  Z_train = z_train,
  s_stream = s_test,
  D_stream = d_test,
  max_iter = 500,
  threshold = 0.5
)

head(stream_fit$posterior)
stream_fit$em$converged
```

Both main functions return posterior probabilities and binary predictions, along with fitted historical summaries. The streaming result also includes EM diagnostics in `stream_fit$em`.

## Evaluation and visualization

The script provides two additional utilities:

- `eval_with_best_threshold(y_true, p_hat)` computes ROC/AUC and classification metrics at a threshold selected using the supplied labels.
- `plot_cv_roc_three_methods(...)` plots fold-specific and mean ROC curves for FIBER, FIBER-Stream, and Fast-DetectGPT. Its posterior inputs are fold-specific matrices, whereas the Fast-DetectGPT input is a full-length score vector.

**Evaluation note:** To avoid information leakage, select decision thresholds using training or validation labels only, then apply the selected thresholds to held-out test data. Do not use held-out labels to tune thresholds.

## Implementation notes

- Binary labels follow the convention `0 = human` and `1 = AI-generated`.
- Detector columns must have the same meaning and order in historical and incoming matrices.
- Detector probabilities use additive (Laplace) smoothing.
- FIBER-Stream performs **batchwise** EM estimation; it is not a fully recursive online algorithm.
- This is the original research implementation with documentation added. In the current code, `fiber_stream()` references `em_fit$params` although `fiber_stream_em()` does not return a `params` field; the estimated means and variances are available directly in `stream_fit$em`.

## Data sources and attribution

The experiments associated with FIBER use data from two previous studies:

1. **Multi-domain detector evaluation and IRT benchmark:** Data and detector-evaluation results from [text-detector-evaluation](https://github.com/YbinZhao/text-detector-evaluation), associated with the work by Zhao et al. (2026), are used as the historical multi-domain source for FIBER. These data support the construction of detector correctness matrices and IRT-based historical summaries.
2. **Fast-DetectGPT data:** The [Fast-DetectGPT repository](https://github.com/baoguangsheng/fast-detect-gpt) and its accompanying datasets (Bao et al., 2024) provide the Fast-DetectGPT benchmark data used for the external out-of-distribution evaluation.

Please consult the original repositories for access to the underlying datasets and their respective usage conditions. The functions in this repository expect detector outputs, Fast-DetectGPT scores, and fitted IRT item parameters to be prepared in advance.

## References and citation

If you use the datasets or detector-evaluation resources, please cite the corresponding original papers:

1. **Zhao, Y., Chen, Z., Bao, G., Zhang, G., and Li, Y.** (2026). *A measurement perspective on benchmarking LLM-generated text detectors using item response theory.* arXiv preprint arXiv:2610.24972. [Paper](https://arxiv.org/abs/2610.24972) | [Data/code](https://github.com/YbinZhao/text-detector-evaluation).
2. **Bao, G., Zhao, Y., Teng, Z., Yang, L., and Zhang, Y.** (2024). *Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature.* In *The Twelfth International Conference on Learning Representations (ICLR 2024).* [Paper](https://openreview.net/forum?id=Bpcgcr8E8Z) | [Code/data](https://github.com/baoguangsheng/fast-detect-gpt).

If you use the FIBER implementation itself, please also cite the corresponding FIBER paper once its bibliographic information is available.
