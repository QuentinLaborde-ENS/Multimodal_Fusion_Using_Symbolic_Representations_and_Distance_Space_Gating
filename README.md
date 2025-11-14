# Multicriteria Semantic Representation of Eye-Tracking Data

Code accompanying the paper: **Multimodal Fusion Using Symbolic Representations and Distance-Space Gating**.  
This repository provides the full, interpretable pipeline for converting heterogeneous physiological and behavioral signals into symbolic sequences, computing modality-specific distances, and fusing them through an adaptive distance-space gating mechanism.

## 🌐 Overview

For each modality (e.g., fixations, saccades, scanpaths, AoIs, ECG, EDA), the pipeline performs:

1. **Feature Normalization (ECDF)**  
   → maps each raw feature to the range **[0, 1]**, using *train-only* distributions.

2. **Adaptive Segmentation (PELT)**  
   → detects piecewise-constant regions in multivariate time series.

3. **Symbolization (Kernel PCA → K-Means)**  
   → embeds segment descriptors in a nonlinear space and assigns each to one of **K symbolic prototypes**.

4. **Sequence Distance (Wagner–Fischer)**  
   → computes a generalized edit distance where substitution costs equal the Euclidean distance between symbol centroids.

5. **Distance-Space Gating**  
   → estimates per-recording modality reliability from local neighbourhood geometry in each modality-specific distance matrix.

6. **Kernel Fusion**  
   → performs symmetric, confidence-weighted bilinear fusion of modality kernels.

7. **Classification**  
   → the fused kernel is fed directly to an SVM with a precomputed kernel.

The goal is to obtain **interpretable, compact, and robust symbolic representations** of multimodal behavioral and physiological signals.

---

 
## ⚙️ Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate multimodal-gating
```


## 🚀 Usage

After installing the dependencies (see [Installation](#-installation)), you can run the main pipeline directly from the command line.

### Example

```bash
python main.py --ternary
```
This will execute the pipeline on the CLDrive dataset, performing ternary task with LOSO cross-validation.

### Command-line Arguments

You can specify the task performed as command-line arguments:
- `--binary` : perform **binary** task with LOSO cross-validation
- `--ternary` : perform **ternary** task with LOSO cross-validation


 



