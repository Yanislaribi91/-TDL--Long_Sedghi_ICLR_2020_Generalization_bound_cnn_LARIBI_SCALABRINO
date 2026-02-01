# -TDL--Long_Sedghi_ICLR_2020_Generalization_bound_cnn_LARIBI_SCALABRINO


# Toy numerical experiments — Long & Sedghi (ICLR 2020)

This repository contains a small, **toy** implementation to reproduce (qualitatively) the type of numerical experiments reported in **Long & Sedghi (ICLR 2020)**, using the spectral/operator norm computation technique from **Sedghi et al. (ICLR 2019)**.

The goal is to study, on a lightweight setup:
- the **distance to initialization**  
  \[
  \|K-K_0\|_{\sigma} \;\stackrel{\mathrm{def}}{=}\; \sum_{i=1}^{L}\left\|\operatorname{op}(K^{(i)}) - \operatorname{op}(K_0^{(i)})\right\|_2
  \]
- the **generalization gap** (here defined as \(|\text{train\_err} - \text{test\_err}|\))
- how these quantities vary with the number of parameters \(W\), and with \(W \cdot \|K-K_0\|_{\sigma}\).

---

## References

- **Sedghi, Gupta, Long (ICLR 2019)** — *The Singular Values of Convolutional Layers*  
  https://arxiv.org/abs/1805.10408
- **Long & Sedghi (ICLR 2020)** — *On the Effect of the Activation Function on the Spectral Bias of Neural Networks* (and related experiments section)  
  (see the paper for the full experimental protocol; this repo focuses on a toy reproduction)

---

## Contents

- `TDL_long_sedghi_iclr_2020_num_exp.ipynb`  
  A self-contained notebook that:
  1. imports dependencies
  2. implements the Sedghi et al. (2019) FFT-based algorithm to compute convolutional operator norms
  3. defines a **small toy CNN** and runs a width sweep
  4. produces plots for:
     - \(\|K-K_0\|_{\sigma}\) vs \(W\)
     - generalization gap vs \(W\)
     - generalization gap vs \(W\cdot\|K-K_0\|_{\sigma}\)

---

## Notebook structure (quick overview)

### 2) Sedghi et al. (2019) operator norm
- `compute_spectral_norm_sedghi(conv_layer, input_spatial_shape)`  
  Computes the spectral norm of the convolution operator using:
  - kernel padding to the input spatial size
  - `fft2` over spatial dimensions
  - batched `svdvals` per frequency
  - taking the maximum singular value over all frequencies

- `compute_distance_from_init(model, initial_model, input_shape)`  
  Iterates over convolution layers, forms the difference of kernels, and sums the corresponding operator norms.

### 3) Model + experiment loop
- `SimpleCNN(width_scale)`  
  A **3-conv** toy CNN (with pooling) for faster runs.
- Dataset: **FashionMNIST**, with a train subset of **5000** samples (for speed)
- Sweep widths: `[1, 2, 4, 8, 16]`
- Training: SGD (`lr=0.01`, `momentum=0.9`) for **5 epochs**
- Metrics:
  - \(W\): total parameter count
  - \(\|K-K_0\|_{\sigma}\): distance to init
  - generalization gap: `abs(train_err - test_err)`
  - \(W\cdot\|K-K_0\|_{\sigma}\)

### 4) Visualization
Three figures are generated:
1. \(\|K-K_0\|_{\sigma}\) vs \(W\)
2. generalization gap vs \(W\)
3. generalization gap vs \(W\cdot\|K-K_0\|_{\sigma}\)



