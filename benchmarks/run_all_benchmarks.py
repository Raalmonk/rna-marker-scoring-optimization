from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    KNOWN_GENE_SYMBOLS,
    PAPER_INTRACELLULAR_MARKERS,
    ExpressionDataset,
    generate_exrna_6cells_benchmark,
    generate_scrna_scaled_tensor,
    load_gse206925,
    simulate_9000_validation_samples,
)
from src.scoring import (
    baseline_scoring_loop,
    baseline_scoring_loop_cell_norm,
    optimized_scoring_convex_numpy,
    optimized_scoring_cpp_csr,
    optimized_scoring_cpp_dense,
    optimized_scoring_cpp_target_csr,
    optimized_scoring_csr_scipy,
    optimized_scoring_target_norm_numpy,
    optimized_scoring_vectorized_prompt,
    pmc10861185_eq1_baseline_loop,
    pmc10861185_eq1_convex_numpy,
    pmc10861185_eq1_cpp_csr,
)
from src.classifiers import evaluate_classifiers_on_9000_samples


def measure_time_and_memory(fn, *args, n_runs: int = 5, measure_mem: bool = True, **kwargs):
    # Warmup
    out = fn(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    median_ms = float(np.median(times))

    peak_mb = 0.0
    if measure_mem:
        tracemalloc.start()
        _ = fn(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024.0 * 1024.0)

    return out, median_ms, peak_mb


def run_part1_gse206925_reproduction(gse: ExpressionDataset) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PART 1: Real GSE206925 Biomarker Reproduction & Mathematical Equivalence Audit")
    print("=" * 80)
    X = gse.rpm
    G, C, R = X.shape
    indptr, data = gse.csr_indptr, gse.csr_data
    print(f"Dataset: {gse.name} | Shape: (G={G}, C={C}, R={R}) | Element Sparsity: {gse.sparsity*100:.2f}% | All-Zero Replicate Blocks: {gse.block_zero_rate*100:.2f}%")

    # 1. Audit Target-Norm (baseline_scoring_loop) vs Cell-Norm (prompt vectorized)
    t0 = time.perf_counter()
    s_base_target = baseline_scoring_loop(X)
    t_base_ms = (time.perf_counter() - t0) * 1000.0

    s_base_cell = baseline_scoring_loop_cell_norm(X)
    s_prompt = optimized_scoring_vectorized_prompt(X)
    s_convex = optimized_scoring_convex_numpy(X)
    s_target_np = optimized_scoring_target_norm_numpy(X)
    s_cpp_csr = optimized_scoring_cpp_csr(indptr, data, G, C, R)
    s_cpp_target_csr = optimized_scoring_cpp_target_csr(indptr, data, G, C, R)

    # Compare prompt vectorized (which uses V / V_max[i]) against baseline_scoring_loop (which uses V / V_max[t])
    audit_discrepancy = {}
    for c_idx, c_name in enumerate(gse.cell_types):
        active = X[:, c_idx, :].mean(axis=1) > 0
        rho_unfixed, _ = spearmanr(s_base_target[active, c_idx], s_prompt[active, c_idx])
        top20_base_t = set(np.argsort(-s_base_target[:, c_idx])[:20])
        top20_prompt = set(np.argsort(-s_prompt[:, c_idx])[:20])
        overlap_unfixed = len(top20_base_t & top20_prompt) / 20.0

        # After fixing normalization consistency:
        rho_fixed_target, _ = spearmanr(s_base_target[active, c_idx], s_cpp_target_csr[active, c_idx])
        top20_fixed_t = set(np.argsort(-s_cpp_target_csr[:, c_idx])[:20])
        overlap_fixed_t = len(top20_base_t & top20_fixed_t) / 20.0

        rho_fixed_cell, _ = spearmanr(s_base_cell[active, c_idx], s_cpp_csr[active, c_idx])
        top20_base_c = set(np.argsort(-s_base_cell[:, c_idx])[:20])
        top20_fixed_c = set(np.argsort(-s_cpp_csr[:, c_idx])[:20])
        overlap_fixed_c = len(top20_base_c & top20_fixed_c) / 20.0

        audit_discrepancy[c_name] = {
            "unfixed_spearman_rho": float(rho_unfixed),
            "unfixed_top20_overlap": float(overlap_unfixed),
            "fixed_target_norm_spearman_rho": float(rho_fixed_target),
            "fixed_target_norm_top20_overlap": float(overlap_fixed_t),
            "fixed_target_norm_max_abs_err": float(np.max(np.abs(s_base_target[:, c_idx] - s_cpp_target_csr[:, c_idx]))),
            "fixed_cell_norm_spearman_rho": float(rho_fixed_cell),
            "fixed_cell_norm_top20_overlap": float(overlap_fixed_c),
            "fixed_cell_norm_max_abs_err": float(np.max(np.abs(s_base_cell[:, c_idx] - s_cpp_csr[:, c_idx]))),
        }
        print(
            f"  [{c_name}] Unfixed Prompt vs Target-Base: rho={rho_unfixed:.6f}, Top20={overlap_unfixed*100:.0f}% | "
            f"Fixed Target-CSR: rho={rho_fixed_target:.8f}, Top20={overlap_fixed_t*100:.0f}%, max_err={audit_discrepancy[c_name]['fixed_target_norm_max_abs_err']:.2e} | "
            f"Fixed Cell-CSR: rho={rho_fixed_cell:.8f}, Top20={overlap_fixed_c*100:.0f}%, max_err={audit_discrepancy[c_name]['fixed_cell_norm_max_abs_err']:.2e}"
        )

    # 2. Verify PMC10861185 Eq. (1) and Figure 1G-I / Figure 2C-D Top Biomarkers on GSE206925
    eq1_loop = pmc10861185_eq1_baseline_loop(X, alpha=0.3)
    eq1_np = pmc10861185_eq1_convex_numpy(X, alpha=0.3)
    eq1_cpp = pmc10861185_eq1_cpp_csr(indptr, data, G, C, R, alpha=0.3)
    finite_mask = np.isfinite(eq1_loop)
    eq1_max_rel_err = float(np.max(np.abs(eq1_loop[finite_mask] - eq1_cpp[finite_mask]) / eq1_loop[finite_mask]))
    print(f"  [PMC10861185 Eq. 1 Additive] Max relative error (Baseline Loop vs C++ CSR): {eq1_max_rel_err:.2e}")

    V = np.mean(X, axis=2)
    std = np.std(X, axis=2)
    marker_verification = []
    print("\n  --- Verification of PMC10861185 Fig. 2C-D Discovered Intracellular Biomarkers ---")
    for c_idx, c_name in enumerate(gse.cell_types):
        for gid, sym in PAPER_INTRACELLULAR_MARKERS[c_name]:
            g_idx = int(np.where(gse.gene_ids == gid)[0][0])
            info = {
                "target_cell": c_name,
                "ensembl_id": gid,
                "symbol": sym,
                "H9_mean_rpm": float(V[g_idx, 0]),
                "NSC_mean_rpm": float(V[g_idx, 1]),
                "EC_mean_rpm": float(V[g_idx, 2]),
                "target_cv": float(std[g_idx, c_idx] / V[g_idx, c_idx]),
                "mult_score": float(s_cpp_csr[g_idx, c_idx]),
                "eq1_score": float(eq1_cpp[g_idx, c_idx]),
            }
            marker_verification.append(info)
            print(
                f"  Target={c_name:3s} | {sym:8s} ({gid}) | "
                f"RPM: H9={info['H9_mean_rpm']:7.2f}, NSC={info['NSC_mean_rpm']:6.2f}, EC={info['EC_mean_rpm']:7.2f} | "
                f"MultScore={info['mult_score']:.5f} | Eq1Score={info['eq1_score']:.2f}"
            )

    return {
        "dataset": gse.name,
        "shape": [G, C, R],
        "sparsity": gse.sparsity,
        "block_zero_rate": gse.block_zero_rate,
        "baseline_loop_ms": t_base_ms,
        "audit_discrepancy": audit_discrepancy,
        "eq1_max_rel_err": eq1_max_rel_err,
        "marker_verification": marker_verification,
    }


def run_part2_speedup_and_memory_scaling(gse: ExpressionDataset) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("PART 2: End-to-End Speedup & Peak Memory Profiling Across Real & scRNA-seq Scales")
    print("=" * 80)

    exrna_6 = generate_exrna_6cells_benchmark(G=27000, R=3, sparsity=0.85)
    scrna_15 = generate_scrna_scaled_tensor(G=20000, C=15, R=50, sparsity=0.90)
    scrna_30 = generate_scrna_scaled_tensor(G=20000, C=30, R=100, sparsity=0.90)
    scrna_60 = generate_scrna_scaled_tensor(G=25000, C=60, R=100, sparsity=0.92)
    scrna_100 = generate_scrna_scaled_tensor(G=25000, C=100, R=200, sparsity=0.95)

    workloads = [
        (gse, True),
        (exrna_6, True),
        (scrna_15, False),
        (scrna_30, False),
        (scrna_60, False),
        (scrna_100, False),
    ]

    scaling_results = []
    for ds, run_full_baseline in workloads:
        X = ds.rpm
        G, C, R = X.shape
        indptr, data = ds.csr_indptr, ds.csr_data
        dense_mb = X.nbytes / (1024.0 * 1024.0)
        csr_mb = (indptr.nbytes + data.nbytes) / (1024.0 * 1024.0)

        if run_full_baseline:
            _, base_ms, base_mem = measure_time_and_memory(
                baseline_scoring_loop_cell_norm, X, n_runs=1, measure_mem=True
            )
            base_estimated = False
        else:
            # Measure on a 500-gene slice and extrapolate linearly by (G / 500)
            G_sub = 500
            X_sub = X[:G_sub]
            _, sub_ms, base_mem = measure_time_and_memory(
                baseline_scoring_loop_cell_norm, X_sub, n_runs=1, measure_mem=True
            )
            base_ms = sub_ms * (G / G_sub)
            base_estimated = True

        _, prompt_ms, prompt_mem = measure_time_and_memory(
            optimized_scoring_vectorized_prompt, X, n_runs=3, measure_mem=True
        )
        _, convex_ms, convex_mem = measure_time_and_memory(
            optimized_scoring_convex_numpy, X, n_runs=5, measure_mem=True
        )
        _, csr_py_ms, csr_py_mem = measure_time_and_memory(
            optimized_scoring_csr_scipy, indptr, data, G, C, R, n_runs=5, measure_mem=True
        )
        _, cpp_dense_ms, _ = measure_time_and_memory(
            optimized_scoring_cpp_dense, X, n_runs=10, measure_mem=False
        )
        _, cpp_csr_ms, _ = measure_time_and_memory(
            optimized_scoring_cpp_csr, indptr, data, G, C, R, n_runs=15, measure_mem=False
        )

        # C++ kernels allocate 2 * G * C * 8 bytes temporary buffer + output G * C * 8 bytes
        cpp_peak_mb = (3 * G * C * 8) / (1024.0 * 1024.0)

        rec = {
            "dataset": ds.name,
            "G": G,
            "C": C,
            "R": R,
            "sparsity": ds.sparsity,
            "block_zero_rate": ds.block_zero_rate,
            "dense_input_mb": dense_mb,
            "csr_input_mb": csr_mb,
            "storage_compression_ratio": dense_mb / csr_mb,
            "baseline_ms": base_ms,
            "baseline_estimated": base_estimated,
            "prompt_vec_ms": prompt_ms,
            "convex_vec_ms": convex_ms,
            "scipy_csr_ms": csr_py_ms,
            "cpp_dense_ms": cpp_dense_ms,
            "cpp_csr_ms": cpp_csr_ms,
            "speedup_prompt": base_ms / prompt_ms,
            "speedup_convex_np": base_ms / convex_ms,
            "speedup_scipy_csr": base_ms / csr_py_ms,
            "speedup_cpp_dense": base_ms / cpp_dense_ms,
            "speedup_cpp_csr": base_ms / cpp_csr_ms,
            "prompt_peak_alloc_mb": prompt_mem,
            "convex_peak_alloc_mb": convex_mem,
            "cpp_peak_alloc_mb": cpp_peak_mb,
            "mem_reduction_convex_vs_prompt": prompt_mem / max(convex_mem, 1e-6),
            "mem_reduction_cpp_vs_prompt": prompt_mem / max(cpp_peak_mb, 1e-6),
        }
        scaling_results.append(rec)

        est_tag = " (extrap.)" if base_estimated else ""
        print(
            f"  {ds.name:24s} | Baseline: {base_ms:9.1f} ms{est_tag} | "
            f"PromptVec: {prompt_ms:6.2f} ms ({rec['speedup_prompt']:6.1f}x) | "
            f"ConvexNP: {convex_ms:6.2f} ms ({rec['speedup_convex_np']:6.1f}x) | "
            f"SciPyCSR: {csr_py_ms:6.2f} ms ({rec['speedup_scipy_csr']:6.1f}x) | "
            f"C++Dense: {cpp_dense_ms:5.2f} ms ({rec['speedup_cpp_dense']:7.1f}x) | "
            f"C++CSR: {cpp_csr_ms:5.2f} ms ({rec['speedup_cpp_csr']:8.1f}x) | "
            f"PeakMem: Prompt={prompt_mem:.1f}MB -> Convex={convex_mem:.1f}MB -> C++={cpp_peak_mb:.1f}MB"
        )

    return scaling_results


def run_part3_underflow_stress_test() -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("PART 3: Numerical Underflow Stress Test (Multiplicative Product vs Log-Domain)")
    print("=" * 80)

    rng = np.random.default_rng(2026)
    G = 1000
    R = 5
    C_list = [6, 15, 30, 50, 80, 120, 200, 300]
    underflow_results = []

    for C in C_list:
        # Background expression in non-target cells with varying expression ratios
        # such that each non-target cell contributes penalty ~ 1 / (1 + 2.5) ~ 0.28
        X = rng.gamma(shape=2.0, scale=40.0, size=(G, C, R))
        # Make top 50 genes strong markers for cell 0
        X[:50, 0, :] = rng.normal(loc=500.0, scale=20.0, size=(50, R))

        V = np.mean(X, axis=2)
        std = np.std(X, axis=2)
        X_max = np.max(X, axis=2)
        X_min = np.min(X, axis=2)
        Z = np.maximum(X_max - V, V - X_min) / np.maximum(std, 1e-12)
        V_max = np.max(V, axis=0, keepdims=True)
        V_norm = V / V_max

        # Compute multiplicative score for t=0 in float32 and float64
        target_term_64 = (V_norm[:, 0]) / (1.0 + Z[:, 0])
        factors_64 = 1.0 / (1.0 + V_norm[:, 1:] * (1.0 + Z[:, 1:]))

        prod_64 = target_term_64.copy()
        for i in range(C - 1):
            prod_64 *= factors_64[:, i]

        prod_32 = target_term_64.astype(np.float32)
        factors_32 = factors_64.astype(np.float32)
        for i in range(C - 1):
            prod_32 = np.float32(prod_32 * factors_32[:, i])

        log_scores = optimized_scoring_convex_numpy(X, return_log=True)[:, 0]

        uf_rate_32 = float(np.mean(prod_32 == 0.0))
        uf_rate_64 = float(np.mean(prod_64 == 0.0))
        uf_rate_log = float(np.mean(~np.isfinite(log_scores)))

        # Rank correlation on the 50 planted markers + background
        rho_32 = float(spearmanr(log_scores, prod_32).statistic) if np.std(prod_32) > 0 else 0.0
        rho_64 = float(spearmanr(log_scores, prod_64).statistic) if np.std(prod_64) > 0 else 0.0

        rec = {
            "C": C,
            "float32_underflow_rate": uf_rate_32,
            "float64_underflow_rate": uf_rate_64,
            "log_domain_underflow_rate": uf_rate_log,
            "float32_spearman_rho": rho_32,
            "float64_spearman_rho": rho_64,
            "min_log10_score": float(np.min(log_scores) / np.log(10.0)),
        }
        underflow_results.append(rec)
        print(
            f"  C={C:3d} | float32 Underflow: {uf_rate_32*100:6.2f}% (rho={rho_32:.4f}) | "
            f"float64 Underflow: {uf_rate_64*100:6.2f}% (rho={rho_64:.4f}) | "
            f"Log-Domain Underflow: {uf_rate_log*100:4.1f}% (min score ~ 10^{rec['min_log10_score']:.0f})"
        )

    return underflow_results


def run_part4_classifier_benchmark(gse: ExpressionDataset) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("PART 4: 9,000-Sample Biomarker Validation (MLP vs GBDT vs Cholesky Shrinkage LDA)")
    print("=" * 80)

    gids = gse.gene_ids
    # 1. MF: Discovered Top-2 Biomarkers per cell type (6 features total)
    mf_ids = [
        "ENSG00000283490",  # MIR518C (H9)
        "ENSG00000207869",  # MIR518B (H9)
        "ENSG00000266782",  # MIR3663 (NSC)
        "ENSG00000283682",  # MIR548AH (NSC)
        "ENSG00000202111",  # VTRNA1-2 (EC)
        "ENSG00000137801",  # THBS1 (EC)
    ]
    mf_idx = [int(np.where(gids == m)[0][0]) for m in mf_ids]

    # 2. C1: Control Set 1 (Moderate-score non-top transcripts that have partial cell expression)
    scores = optimized_scoring_cpp_csr(
        gse.csr_indptr, gse.csr_data, *gse.shape
    )
    c1_idx = []
    for c in range(3):
        ranked = np.argsort(-scores[:, c])
        # Pick ranks 40 and 41 for each cell type
        c1_idx.extend([int(ranked[40]), int(ranked[41])])

    # 3. C2: Control Set 2 (High-abundance house-keeping / non-specific RNAs like MIR302A/B, MIR148A, or random low-score genes)
    rng = np.random.default_rng(42)
    zero_or_nonspec = np.where((gse.rpm.mean(axis=(1, 2)) > 0.1) & (scores.max(axis=1) < 1e-4))[0]
    c2_idx = [int(x) for x in rng.choice(zero_or_nonspec, size=6, replace=False)]

    all_clf_records = []
    for fset_name, f_indices in [("MF (Top Biomarkers)", mf_idx), ("C1 (Moderate Control)", c1_idx), ("C2 (Random Non-Markers)", c2_idx)]:
        X_tr, y_tr, X_te, y_te = simulate_9000_validation_samples(gse, f_indices, noise_std=0.75, seed=42)
        res_list = evaluate_classifiers_on_9000_samples(
            X_tr, y_tr, X_te, y_te, gse.cell_types, feature_set_name=fset_name
        )
        print(f"\n  --- Feature Set: {fset_name} (7,200 train / 1,800 test) ---")
        for r in res_list:
            rec = {
                "feature_set": r.feature_set,
                "model_name": r.model_name,
                "train_time_ms": r.train_time_ms,
                "infer_time_ms": r.infer_time_ms,
                "total_time_ms": r.total_time_ms,
                "speedup_vs_mlp_grid": r.speedup_vs_mlp_grid,
                "speedup_vs_mlp_single": r.speedup_vs_mlp_single,
                "accuracy": r.accuracy,
                "macro_roc_auc": r.macro_roc_auc,
                "macro_pr_auc": r.macro_pr_auc,
                "per_cell_roc_auc": r.per_cell_roc_auc,
                "per_cell_pr_auc": r.per_cell_pr_auc,
            }
            all_clf_records.append(rec)
            print(
                f"  {r.model_name:42s} | Train: {r.train_time_ms:8.2f} ms | Infer: {r.infer_time_ms:5.2f} ms | "
                f"Speedup vs GridMLP: {r.speedup_vs_mlp_grid:8.1f}x | Acc: {r.accuracy*100:5.2f}% | "
                f"ROC-AUC: {r.macro_roc_auc:.4f} | PR-AUC: {r.macro_pr_auc:.4f}"
            )

    return all_clf_records


def generate_summary_plots(
    scaling_results: List[Dict[str, Any]],
    underflow_results: List[Dict[str, Any]],
    clf_results: List[Dict[str, Any]],
    output_path: str = "results/benchmark_summary.png",
):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)

    # Panel A: End-to-End Speedup Across Datasets (Log scale)
    ax = axes[0, 0]
    labels = [r["dataset"].replace("PMC10861185_", "").replace("_3cells", "") for r in scaling_results]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, [r["speedup_convex_np"] for r in scaling_results], width, label="Convex 1-Pass NumPy", color="#3b82f6")
    ax.bar(x, [r["speedup_cpp_dense"] for r in scaling_results], width, label="C++17 NEON/GCD Dense", color="#10b981")
    ax.bar(x + width, [r["speedup_cpp_csr"] for r in scaling_results], width, label="C++17 NEON/GCD CSR (Zero-Skip)", color="#f59e0b")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Speedup vs. Baseline O(C²GR) Loop (log scale)")
    ax.set_title("(A) End-to-End Scoring Speedup Across Data Scales")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel B: Peak Memory Consumption (3D Broadcast vs Convex 1-Pass vs C++ CSR)
    ax = axes[0, 1]
    ax.bar(x - width, [r["prompt_peak_alloc_mb"] for r in scaling_results], width, label="3D Broadcast Vectorized", color="#ef4444")
    ax.bar(x, [r["convex_peak_alloc_mb"] for r in scaling_results], width, label="Convex 1-Pass (Zero 3D Tensor)", color="#3b82f6")
    ax.bar(x + width, [r["cpp_peak_alloc_mb"] for r in scaling_results], width, label="Fused C++17 CSR Kernel", color="#10b981")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Peak Auxiliary Memory Allocated (MB, log scale)")
    ax.set_title("(B) Peak Memory Reduction via Convex Identity & Fused Kernels")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel C: Numerical Underflow Rate vs Number of Cell Types C
    ax = axes[1, 0]
    C_vals = [r["C"] for r in underflow_results]
    ax.plot(C_vals, [r["float32_underflow_rate"] * 100 for r in underflow_results], "o-", linewidth=2, color="#ef4444", label="Baseline Multiplicative (float32)")
    ax.plot(C_vals, [r["float64_underflow_rate"] * 100 for r in underflow_results], "s--", linewidth=2, color="#f97316", label="Baseline Multiplicative (float64)")
    ax.plot(C_vals, [r["log_domain_underflow_rate"] * 100 for r in underflow_results], "^-", linewidth=2.5, color="#10b981", label="Log-Domain Prefix-Sum (Ours)")
    ax.set_xlabel("Number of Cell Types / Single-Cell Clusters (C)")
    ax.set_ylabel("Underflow Rate (% Flushed to 0.0)")
    ax.set_title("(C) IEEE-754 Numerical Underflow vs. Cell Cluster Count C")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center right", fontsize=8.5)

    # Panel D: 9,000-Sample Classifier Latency vs ROC-AUC on GSE206925
    ax = axes[1, 1]
    mf_clfs = [r for r in clf_results if r["feature_set"].startswith("MF")]
    names = [
        "MLP (GridCV 3-Fold)",
        "MLP (Single 3x3 Id)",
        "HistGBDT",
        "Sklearn LDA",
        "Cholesky LDA (Ours)",
    ]
    times = [r["total_time_ms"] for r in mf_clfs]
    rocs = [r["macro_roc_auc"] for r in mf_clfs]
    colors = ["#ef4444", "#f97316", "#8b5cf6", "#3b82f6", "#10b981"]

    bars = ax.barh(names, times, color=colors, height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Total Training + Inference Time on 9,000 Samples (ms, log scale)")
    ax.set_title("(D) 9,000-Sample Validator Latency & AUROC (GSE206925 MF)")
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.4)
    for bar, roc, t in zip(bars, rocs, times):
        ax.text(
            t * 1.15,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f} ms (AUC={roc:.3f})",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0.1, max(times) * 18)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"\nSaved 4-panel summary figure to: {output_path}")


def main():
    gse = load_gse206925()
    part1 = run_part1_gse206925_reproduction(gse)
    part2 = run_part2_speedup_and_memory_scaling(gse)
    part3 = run_part3_underflow_stress_test()
    part4 = run_part4_classifier_benchmark(gse)

    generate_summary_plots(part2, part3, part4, output_path="results/benchmark_summary.png")

    os.makedirs("results", exist_ok=True)
    full_report = {
        "part1_gse206925_reproduction": part1,
        "part2_speedup_and_memory_scaling": part2,
        "part3_underflow_stress_test": part3,
        "part4_classifier_9000_samples": part4,
    }
    with open("results/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print("Saved full quantitative benchmark data to: results/benchmark_results.json")


if __name__ == "__main__":
    main()
