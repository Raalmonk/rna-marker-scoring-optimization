# High-Performance Statistical Scoring & Biomarker Discovery (`PMC10861185` / `GSE206925`)

End-to-end reproduction, mathematical audit, and systems/algorithmic optimization of the RNA cell-type biomarker discovery framework by **Shi et al. (Yuguo Lei & Chi Zhang Lab)**, *Gene Reports*, Vol. 34, 101869 (2024) (`PMC10861185` / `PMID: 38351912`, NCBI GEO **`GSE206925`**).

---

## 1. Key Findings & Optimization Summary

### A. Real `GSE206925` Biomarker Reproduction & Mathematical Equivalence Audit
1. **100% Exact Biomarker Reproduction on `GSE206925` ($G = 60,675$ transcripts, $C = 3$ cell types, $R = 3$ biological replicates, $80.11\%$ element sparsity, $71.56\%$ all-zero replicate blocks)**:
   - **H9 (hESC)**: `MIR518C` (`ENSG00000283490`, $226.57\text{ RPM}$) & `MIR518B` (`ENSG00000207869`, $170.71\text{ RPM}$)
   - **NSC (Neural Stem Cells)**: `MIR3663` (`ENSG00000266782`, $53.19\text{ RPM}$) & `MIR548AH` (`ENSG00000283682`, $33.67\text{ RPM}$)
   - **EC (Endothelial Cells)**: `VTRNA1-2` (`ENSG00000202111`, $1065.32\text{ RPM}$) & `THBS1` (`ENSG00000137801`, $777.84\text{ RPM}$)
2. **Resolution of Normalization Discrepancy**:
   - In the baseline nested loop (`baseline_scoring_loop`), non-target cell mean expression $v_{g,i}$ was normalized by the *target* cell's global maximum $V_{\max, t}$, whereas naive 2-stage vectorization (`V_norm = V / V_max`) normalized by $V_{\max, i}$. Because $V_{\max,\text{H9}} = 403,952\text{ RPM}$ while $V_{\max,\text{EC}} = 41,903\text{ RPM}$ ($9.6\times$ difference), unfixed vectorization yielded a Top-20 overlap of $95\%$ on H9 and EC.
   - We provide exact two-stage kernels for **both** cell-normalized ($v_{g,i}/V_{\max,i}$, matching `PMC10861185` Eq. 1) and target-normalized ($v_{g,i}/V_{\max,t}$) formulations, achieving **Spearman $\rho = 1.00000000$, Top-20 overlap $= 100\%$, and maximum numerical error $\le 6.52 \times 10^{-16}$ (machine epsilon)**.

### B. Four-Stage Algorithmic & Hardware Optimization Pipeline
1. **Optimization 1 — Two-Stage Computation Graph & Prefix-Sum Elimination ($O(C^2 G R) \to O(C G R)$)**:
   - Decouples replicate reduction from cross-cell-type aggregation via $\sum_{i \neq t} \phi(v_{g,i}, Z_{g,i}) = \left(\sum_{i=1}^C \phi(v_{g,i}, Z_{g,i})\right) - \phi(v_{g,t}, Z_{g,t})$.
2. **Optimization 2 — Convex Extreme-Point Z-Score Identity & Log-Domain Branchless Vectorization**:
   - **Convex Extreme-Point Identity**: By exploiting the convexity of $|\cdot|$, we prove that:
     $$\max_{r \in \{1,\dots,R\}} |X_{g,c,r} - v_{g,c}| \equiv \max\left(\max_r X_{g,c,r} - v_{g,c},\; v_{g,c} - \min_r X_{g,c,r}\right)$$
     holds with **$0.0$ IEEE-754 bit error**, eliminating the 3D intermediate broadcast tensor `X - V[:, :, None]` of shape $(G, C, R)$ and reducing peak memory allocation by **$2\times \text{ to } 134\times$**.
   - **Log-Domain Transformation**: Converts the $(C-1)$-term product into additive log-penalties, preventing `float32` / `float64` flush-to-zero underflow at large $C$.
3. **Optimization 3 — CSR Sparse Zero-Skipping + Fused C++17 ARM NEON / Apple GCD Multi-Core Kernel ($O(C G R) \to O(\text{nnz})$)**:
   - Stores replicate blocks in CSR format: skips all-zero blocks ($\text{nnz}_{g,c} = 0$, $71.6\%\text{–}95\%$ of blocks) in $O(1)$ and computes centered variance algebraically via $(R - \text{nnz}_{g,c}) v_{g,c}^2 + \sum_{x \neq 0} (x - v_{g,c})^2$.
   - Achieving **$345\times$ speedup on `GSE206925`**, **$1,306\times$ speedup on 6-cell ex-RNA**, and **$37,152\times \text{ to } 76,957\times$ speedup** on single-cell RNA-seq scales ($C=30\text{–}100$).
4. **Optimization 4 — Closed-Form Cholesky Shrinkage LDA Validator ($>14,900\times$ Speedup on 9,000 Samples)**:
   - In `PMC10861185` Section 2.6, 3-fold CV grid search selected a 5-layer MLP with linear `activation='identity'` and 3 hidden units per layer trained with iterative Adam backprop. Algebraically, an identity-activated MLP collapses to a linear discriminant projection.
   - Replacing iterative `MLPClassifier` with our **Closed-Form Cholesky Shrinkage LDA** compresses 9,000-sample training + validation latency from **$13,699.7\text{ ms}$ (GridSearchCV MLP) / $493.8\text{ ms}$ (Single MLP) down to $0.92\text{ ms}$ ($14,904\times$ / $537\times$ speedup)** while achieving **$99.33\%$ accuracy and $0.9997$ ROC-AUC**.

---

## 2. Repository Structure

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

---

## 3. Quick Start

```bash
# 1. Compile the C++17 ARM NEON / GCD shared library
clang++ -O3 -std=c++17 -mcpu=apple-m1 -ffast-math -shared -fPIC src/fast_kernels.cpp -o src/libfast_kernels.dylib

# 2. Run numerical equivalence and unit tests
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

# 3. Run the complete benchmark suite and regenerate figures
.venv/bin/python benchmarks/run_all_benchmarks.py
```
