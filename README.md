# Symbolic Representations and Geometry-Driven Gating for Physiological State Inference

Code accompanying the paper: **Beyond Deep Fusion: Symbolic Representations and Geometry-Driven Gating for Physiological State Inference**.  
This repository provides the full, interpretable pipeline for converting heterogeneous physiological and behavioral signals into symbolic sequences, computing modality-specific distances, and fusing them through an adaptive distance-space gating mechanism.

## 🌐 Overview – The Pipeline at a Glance

The method turns raw, heterogeneous, and noisy physiological/behavioral signals (eye fixations, saccades, scanpaths, areas of interest, ECG, EDA) into **compact, interpretable symbolic sequences** and fuses them in a fully distance-based, adaptive way — no deep networks, no attention, no feature-level concatenation.

For **every modality**, the **first five steps** are applied independently to generate modality-specific symbolic sequences and distance matrices:

1. **Feature Normalization (ECDF)**  
   Empirical cumulative distribution mapping → all features scaled to [0, 1] using **training-set statistics only** (perfect subject-invariance).

2. **Adaptive Segmentation (PELT)**  
   Change-point detection that automatically splits each recording into quasi-stationary regimes (Pruned Exact Linear Time algorithm).

3. **Nonlinear Symbolization (Kernel PCA + K-Means)**  
   Segment means are embedded via RBF Kernel PCA (captures nonlinear structure), then clustered into **K = 20 symbolic prototypes** per modality. Each prototype is an interpretable "physiological/behavioral motif".

4. **Learned Edit Distance (Wagner–Fischer)**  
   Pairwise dissimilarity between two recordings = generalized Levenshtein distance on their symbolic sequences, where substitution cost = Euclidean distance between prototype centroids in kernel space.

5. **Distance-Space Gating (ultra-lightweight)**  
   Three scale-free geometric indicators (local density, regularity, label purity) are extracted from each modality's distance matrix → one **regularized logistic regression per modality** (≤ 30 parameters total) predicts per-sample reliability in a self-supervised way.

**Then, across all modalities** (steps 6–7):  
6. **Confidence-Weighted Kernel Fusion**  
   Modality-specific distance matrices → RBF kernels → symmetric pairwise gating → final positive-definite fused kernel (localized multiple kernel learning style, but with almost zero trainable parameters).

7. **Classification**  
   The fused kernel is directly fed to a standard SVM (precomputed-kernel mode). No further training needed.

**Result:** a fully symbolic, interpretable, data-efficient, and highly robust multimodal classifier that consistently outperforms deep-learning baselines on the CL-Drive cognitive-load benchmark — even without using EEG.

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


 



