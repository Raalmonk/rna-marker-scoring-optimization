# High-Performance Statistical Scoring & Biomarker Discovery (`PMC10861185` / `GSE206925`)

End-to-end reproduction, mathematical audit, and four-stage systems/algorithmic optimization of the RNA cell-type biomarker discovery framework by **Shi et al. (Yuguo Lei & Chi Zhang Lab)**, *Identification of RNA-based cell-type markers for stem-cell manufacturing systems with a statistical scoring function*, *Gene Reports*, Volume 34, 101869 (2024) (**`PMC10861185`** / `PMID: 38351912`, NCBI GEO **`GSE206925`**).

While preserving **100% mathematical and rank-order equivalence** (**Spearman $\rho = 1.00000000$**, **Top-20 biomarker overlap $= 100\%$**, and **maximum floating-point deviation $\le 6.52 \times 10^{-16}$** across all $60,675$ transcripts):
- **Statistical Scoring Function (Eq. 1) Speedup**: Achieves a **$345.0\times$** speedup on the real `GSE206925` smRNA-seq dataset, **$1,306.5\times$** on the 6-cell-line ex-RNA dataset, and **$11,618\times \text{ to } 76,957\times$** on single-cell RNA-seq scales ($C=15\text{–}100$) — outperforming standard NumPy vectorization by an additional **$1.5\times \text{ to } 478.3\times$** while reducing peak memory consumption by **$4.2\times \text{ to } 134.4\times$**.
- **9,000-Sample Biomarker Validation Speedup**: Replaces the iterative 5-layer identity-activated Multi-Layer Perceptron (`MLPClassifier`) with a **Closed-Form Cholesky Shrinkage Linear Discriminant Analysis (`CholeskyShrinkageLDA`)** solver, compressing training + inference latency on 9,000 simulated samples from **$13,699.67\text{ ms}$ (3-fold GridSearchCV MLP) / $493.81\text{ ms}$ (single MLP) down to $0.92\text{ ms}$** (**$14,903.8\times$ / $536.8\times$ speedup**) while achieving **$0.9997$ ROC-AUC** and **$99.33\%$ test accuracy**.

![Benchmark Summary Across All Four Optimization Dimensions](results/benchmark_summary.png)

---

## 1. Quantitative Experimental Results

### A. End-to-End Scoring Latency, Speedup & Peak Memory Scaling (Experiment 2)

Evaluated on Apple M1 Pro (10 Cores, 16 GB RAM) in IEEE-754 `float64` precision across real `GSE206925`, `PMC10861185` 6-cell ex-RNA, and multi-cluster scRNA-seq workloads:

| Dataset & Dimensions $(G, C, R)$ | Zero Sparsity | Baseline Loop $O(C^2GR)$ | Standard Vectorized NumPy (3D Broadcast) | Convex 1-Pass NumPy (Zero 3D Tensor) | SciPy CSR Zero-Skipping | C++17 NEON/GCD Dense Fused | **C++17 NEON/GCD CSR Fused (Ours)** | **Speedup vs. Baseline** | **Speedup vs. Std. NumPy** | **Peak Memory Reduction (3D Broadcast $\to$ Ours)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`GSE206925` Real smRNA** $(60675, 3, 3)$ | $80.1\%$ | $2,661.3\text{ ms}$ | $11.84\text{ ms}$ | $13.06\text{ ms}$ | $7.70\text{ ms}$ | $8.81\text{ ms}$ | **$7.71\text{ ms}$** | **$345.0\times$** | $1.54\times$ | $17.5\text{ MB} \to \mathbf{4.2\text{ MB}}$ ($4.2\times$) |
| **`PMC10861185` ex-RNA** $(27000, 6, 3)$ | $85.0\%$ | $4,350.4\text{ ms}$ | $10.97\text{ ms}$ | $11.99\text{ ms}$ | $6.61\text{ ms}$ | $3.81\text{ ms}$ | **$3.33\text{ ms}$** | **$1,306.5\times$** | $3.29\times$ | $15.3\text{ MB} \to \mathbf{3.7\text{ MB}}$ ($4.1\times$) |
| **scRNA Small** $(20000, 15, 50)$ | $90.0\%$ | $45,158.9\text{ ms}$ | $58.72\text{ ms}$ | $46.75\text{ ms}$ | $28.77\text{ ms}$ | $5.03\text{ ms}$ | **$3.89\text{ ms}$** | **$11,618.1\times$** | $15.1\times$ | $236.1\text{ MB} \to \mathbf{6.9\text{ MB}}$ ($34.2\times$) |
| **scRNA Medium** $(20000, 30, 100)$ | $90.0\%$ | $178,196.8\text{ ms}$ | $171.01\text{ ms}$ | $122.61\text{ ms}$ | $62.15\text{ ms}$ | $8.82\text{ ms}$ | **$4.80\text{ ms}$** | **$37,152.4\times$** | $35.6\times$ | $929.9\text{ MB} \to \mathbf{13.7\text{ MB}}$ ($67.9\times$) |
| **scRNA Large** $(25000, 60, 100)$ | $92.0\%$ | $791,861.4\text{ ms}$ | $2,457.04\text{ ms}$ | $309.07\text{ ms}$ | $145.39\text{ ms}$ | $20.10\text{ ms}$ | **$10.29\text{ ms}$** | **$76,957.0\times$** | **$238.8\times$** | $2,324.6\text{ MB} \to \mathbf{34.3\text{ MB}}$ ($67.8\times$) |
| **scRNA Ultra** $(25000, 100, 200)$ | $95.0\%$ | $1,747,892.4\text{ ms}$ | $16,945.11\text{ ms}$ | $5,878.36\text{ ms}$ | $272.99\text{ ms}$ | $2,002.53\text{ ms}$ | **$35.43\text{ ms}$** | **$49,338.3\times$** | **$478.3\times$** | $7,689.1\text{ MB} \to \mathbf{57.2\text{ MB}}$ ($134.4\times$) |

---

### B. Real `GSE206925` Biomarker Reproduction & Numerical Equivalence Audit (Experiment 1)

On the real NCBI GEO `GSE206925` dataset ($G = 60,675$ human Ensembl transcripts across `H9`, `NSC`, and `EC`, $R = 3$ biological replicates per cell type, $80.11\%$ element sparsity, $71.56\%$ all-zero replicate blocks):

| Target Cell Type | Unfixed Vectorized vs. Target-Base ($\rho$ / Top-20) | **Fixed Target-CSR Kernel ($\rho$ / Top-20 / Max Abs Error)** | **Fixed Cell-CSR Kernel ($\rho$ / Top-20 / Max Abs Error)** | **Exact `PMC10861185` Eq.(1) CSR Kernel (Max Rel Error)** |
| :--- | :---: | :---: | :---: | :---: |
| **H9 (hESC)** | `0.999995` / `95%` | **`1.00000000` / `100%` / `1.39e-16`** | **`1.00000000` / `100%` / `1.94e-16`** | **`6.52e-16`** |
| **NSC (Neural Stem Cells)** | `0.999995` / `100%` | **`1.00000000` / `100%` / `1.53e-16`** | **`1.00000000` / `100%` / `1.67e-16`** | **`6.52e-16`** |
| **EC (Endothelial Cells)** | `1.000000` / `95%` | **`1.00000000` / `100%` / `1.11e-16`** | **`1.00000000` / `100%` / `1.11e-16`** | **`6.52e-16`** |

All 6 intracellular biomarkers reported in **Figure 2C–D** of `PMC10861185` are reproduced identically:

| Target Lineage | Gene Symbol | Ensembl ID | H9 Mean (RPM) | NSC Mean (RPM) | EC Mean (RPM) | Target CV ($\sigma_t / v_t$) | Multiplicative Score $S_{\text{mult}}$ | Additive Eq.(1) Score $S_{\text{Eq1}}$ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **H9 (hESC)** | **`MIR518C`** | `ENSG00000283490` | **226.57** | 16.72 | 0.56 | 0.083 | `0.00023` | `1784.94` |
| **H9 (hESC)** | **`MIR518B`** | `ENSG00000207869` | **170.71** | 7.02 | 0.54 | 0.311 | `0.00018` | `2368.23` |
| **NSC** | **`MIR3663`** | `ENSG00000266782` | 0.27 | **53.19** | 0.00 | 0.181 | `0.00006` | `8157.66` |
| **NSC** | **`MIR548AH`** | `ENSG00000283682` | 5.53 | **33.67** | 0.00 | 0.076 | `0.00003` | `12887.69` |
| **EC** | **`VTRNA1-2`** | `ENSG00000202111` | 1.22 | 2.49 | **1065.32** | 1.064 | `0.00269` | `156.50` |
| **EC** | **`THBS1`** | `ENSG00000137801` | 0.30 | 0.18 | **777.84** | 0.936 | `0.00196` | `213.43` |

---

### C. IEEE-754 Numerical Underflow Stress Test Across Cluster Count $C$ (Experiment 3)

As the number of cell types / single-cell clusters $C$ scales from $6$ to $300$, multiplicative accumulation in `float32` suffers from rank collapse ($\rho = 0.0000$) at $C \ge 80$ and $100\%$ flush-to-zero underflow at $C \ge 200$, whereas our log-domain prefix-sum kernel maintains **$0.0\%$ underflow** across all scales:

| Cell Types / Clusters $C$ | Multiplicative `float32` Underflow Rate | `float32` Rank Correlation $\rho$ | **Log-Domain Underflow Rate (Ours)** | Minimum Score Magnitude ($\log_{10} S_{\min}$) |
| :---: | :---: | :---: | :---: | :---: |
| $C = 6$ | $0.00\%$ | `1.0000` | **$0.00\%$** | $10^{-4}$ |
| $C = 30$ | $0.00\%$ | `1.0000` | **$0.00\%$** | $10^{-12}$ |
| $C = 50$ | $0.00\%$ | `1.0000` | **$0.00\%$** | $10^{-19}$ |
| $C = 80$ | $0.00\%$ (top markers flushed) | **`0.0000` (rank collapse)** | **$0.00\%$** | $10^{-29}$ |
| $C = 120$ | $0.00\%$ (top markers flushed) | **`0.0000` (rank collapse)** | **$0.00\%$** | $10^{-42}$ |
| $C = 200$ | **$100.00\%$ (total flush to 0.0)** | **`0.0000` (rank collapse)** | **$0.00\%$** | $10^{-68}$ |
| $C = 300$ | **$100.00\%$ (total flush to 0.0)** | **`0.0000` (rank collapse)** | **$0.00\%$** | $10^{-101}$ |

---

### D. 9,000-Sample Biomarker Validation Benchmark on `GSE206925` (Experiment 4)

Following `PMC10861185` Section 2.6–2.7, 9,000 synthetic transcriptomes ($7,200$ training + $1,800$ independent test samples, $\sigma_{\text{noise}} = 0.75$) were generated from the 9 `GSE206925` biological samples to evaluate the Marker Feature Set (`MF`), Control Set 1 (`C1`, moderate-score transcripts), and Control Set 2 (`C2`, random non-marker transcripts):

| Feature Set | Classifier Model | Train Latency | Inference Latency | Total Latency | **Speedup vs. GridMLP** | **Speedup vs. Single MLP** | Test Accuracy | Macro ROC-AUC | Macro PR-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MF (Top Biomarkers)** | `MLP (Paper GridSearchCV 3-Fold)` | $13,697.93\text{ ms}$ | $1.74\text{ ms}$ | $13,699.67\text{ ms}$ | $1.0\times$ | $0.04\times$ | $99.61\%$ | `0.9999` | `0.9997` |
| **MF (Top Biomarkers)** | `MLP (Paper 3x3 Identity, Adam)` | $493.27\text{ ms}$ | $0.54\text{ ms}$ | $493.81\text{ ms}$ | $27.7\times$ | $1.0\times$ | $99.17\%$ | `0.9997` | `0.9994` |
| **MF (Top Biomarkers)** | `HistGradientBoosting (GBDT)` | $830.27\text{ ms}$ | $12.03\text{ ms}$ | $842.30\text{ ms}$ | $16.3\times$ | $0.59\times$ | $99.61\%$ | `1.0000` | `0.9999` |
| **MF (Top Biomarkers)** | `Sklearn Shrinkage LDA` | $4.48\text{ ms}$ | $0.39\text{ ms}$ | $4.87\text{ ms}$ | $2,813.5\times$ | $101.4\times$ | $99.33\%$ | `0.9997` | `0.9995` |
| **MF (Top Biomarkers)** | **`Cholesky Shrinkage LDA (Ours)`** | **$0.72\text{ ms}$** | **$0.20\text{ ms}$** | **$0.92\text{ ms}$** | **$14,903.8\times$** | **$536.8\times$** | **$99.33\%$** | **`0.9997`** | **`0.9995`** |
| **C1 (Moderate Control)** | `MLP (Paper GridSearchCV 3-Fold)` | $12,845.09\text{ ms}$ | $0.76\text{ ms}$ | $12,845.85\text{ ms}$ | $1.0\times$ | $0.07\times$ | $69.61\%$ | `0.8573` | `0.7211` |
| **C1 (Moderate Control)** | **`Cholesky Shrinkage LDA (Ours)`** | **$0.64\text{ ms}$** | **$0.19\text{ ms}$** | **$0.83\text{ ms}$** | **$15,503.8\times$** | **$1,142.6\times$** | **$68.44\%$** | **`0.8578`** | **`0.7228`** |
| **C2 (Random Control)** | `MLP (Paper GridSearchCV 3-Fold)` | $10,897.21\text{ ms}$ | $0.84\text{ ms}$ | $10,898.05\text{ ms}$ | $1.0\times$ | $0.07\times$ | $56.50\%$ | `0.7407` | `0.5975` |
| **C2 (Random Control)** | **`Cholesky Shrinkage LDA (Ours)`** | **$0.77\text{ ms}$** | **$0.22\text{ ms}$** | **$0.99\text{ ms}$** | **$11,043.2\times$** | **$807.9\times$** | **$52.44\%$** | **`0.7188`** | **`0.5676`** |

---

## 2. Four-Stage Algorithmic & Systems Innovations

1. **Two-Stage Computation Graph & Prefix-Sum Elimination ($O(C^2 G R) \to O(C G R)$)**:
   Decouples replicate reduction from cross-cell-type aggregation via global row-sum elimination in the log domain:
   $$\sum_{i \neq t} \phi(v_{g,i}, Z_{g,i}) = \left(\sum_{i=1}^C \phi(v_{g,i}, Z_{g,i})\right) - \phi(v_{g,t}, Z_{g,t})$$
   reducing Stage 2 synthesis complexity from $O(G C^2)$ to $O(G C)$.
2. **Convex Extreme-Point Single-Pass Z-Score Identity ($0.0$ Bit Error & $134\times$ Lower Peak Memory)**:
   Standard vectorized implementations (`np.max(np.abs(X - V[:, :, None]) / safe_std[:, :, None], axis=2)`) allocate three 3D intermediate tensors of shape $(G, C, R)$ ($7.69\text{ GB}$ at $25000 \times 100 \times 200$). By exploiting the convexity of $f(x) = |x - v_{g,c}|$ over $[X_{\min}, X_{\max}]$:
   $$\max_{1 \le r \le R} |X_{g,c,r} - v_{g,c}| \equiv \max\left(\max_{r} X_{g,c,r} - v_{g,c},\; v_{g,c} - \min_{r} X_{g,c,r}\right)$$
   we eliminate all 3D broadcast tensors with **$0.0$ IEEE-754 bit error** and enable single-pass register reduction.
3. **CSR Sparse Block Zero-Skipping + Fused C++17 ARM NEON / Apple GCD Kernel ($O(C G R) \to O(\text{nnz})$)**:
   Stores replicate blocks in Compressed Sparse Row (CSR) format (`indptr`, `data`). All-zero blocks ($\text{nnz}_{g,c} = 0$, $71.6\%\text{–}95\%$ of blocks) are skipped in $O(1)$, and partially non-zero blocks compute exact centered variance algebraically via $(R - \text{nnz}_{g,c})v_{g,c}^2 + \sum_{x \neq 0}(x - v_{g,c})^2$ across all 10 Apple M1 Pro cores.
4. **Replacing the 5-Layer Identity-Activated MLP with Closed-Form Cholesky Shrinkage LDA**:
   In `PMC10861185` Section 2.6, 3-fold CV grid search selected a 5-layer MLP with `activation='identity'` (linear activation) and 3 hidden units per layer. Algebraically, a multi-layer network with identity activations collapses to a single linear projection $\mathbf{W}_{\text{eff}}\mathbf{x} + \mathbf{b}_{\text{eff}}$. Solving the exact regularized linear discriminant in closed form via Cholesky decomposition ($\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Sigma}_\lambda$) eliminates iterative Adam backpropagation and achieves a **$14,904\times$ speedup**.

---

## 3. Repository Structure & Quick Start

```text
.
├── data/
│   └── raw/                        # Real NCBI GEO GSE206925 count matrices (GSM6266913–GSM6266921)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # GSE206925 loader, CSR block builder, scRNA & 9,000-sample simulator
│   ├── fast_kernels.cpp            # C++17 ARM NEON + Apple GCD (10-core) fused dense & CSR kernels
│   ├── libfast_kernels.dylib       # Compiled native shared library
│   ├── scoring.py                  # Baseline & optimized scoring functions (Eq. 1 & Multiplicative)
│   └── classifiers.py              # Closed-Form Cholesky Shrinkage LDA vs. MLP / GBDT validators
├── benchmarks/
│   └── run_all_benchmarks.py       # End-to-end benchmark runner & 4-panel figure generator
├── tests/
│   └── test_all.py                 # Unit tests for mathematical identities & 1e-16 equivalence
└── results/
    ├── benchmark_summary.png       # 4-panel quantitative performance & accuracy figure
    └── benchmark_results.json      # Full machine-readable benchmark metrics
```

### Run Tests & Reproduce All Benchmarks

```bash
# 1. Compile the C++17 ARM NEON / GCD shared library
clang++ -O3 -std=c++17 -mcpu=apple-m1 -ffast-math -shared -fPIC src/fast_kernels.cpp -o src/libfast_kernels.dylib

# 2. Run numerical equivalence and unit tests (6/6 tests)
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

# 3. Run the complete benchmark suite and regenerate figures & JSON metrics
.venv/bin/python benchmarks/run_all_benchmarks.py
```
