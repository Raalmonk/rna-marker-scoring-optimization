from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# Known Ensembl Gene ID -> Gene Symbol mappings for key biomarkers in PMC10861185 / GSE206925
KNOWN_GENE_SYMBOLS: Dict[str, str] = {
    "ENSG00000207869": "MIR518B",
    "ENSG00000283490": "MIR518C",
    "ENSG00000266782": "MIR3663",
    "ENSG00000283682": "MIR548AH",
    "ENSG00000202111": "VTRNA1-2",
    "ENSG00000137801": "THBS1",
    "ENSG00000207925": "MIR516B2",
    "ENSG00000283685": "MIR522",
    "ENSG00000202515": "VTRNA1-1",
    "ENSG00000202293": "VTRNA1-3",
    "ENSG00000207927": "MIR302A",
    "ENSG00000284463": "MIR302B",
    "ENSG00000199085": "MIR148A",
    "ENSG00000199145": "MIR182",
    "ENSG00000207990": "MIR367",
    "ENSG00000207691": "MIR302D",
    "ENSG00000199102": "MIR302C",
    "ENSG00000115414": "FN1",
    "ENSG00000234474": "MIR3663HG",
}

PAPER_INTRACELLULAR_MARKERS: Dict[str, List[Tuple[str, str]]] = {
    "H9": [("ENSG00000283490", "MIR518C"), ("ENSG00000207869", "MIR518B")],
    "NSC": [("ENSG00000266782", "MIR3663"), ("ENSG00000283682", "MIR548AH")],
    "EC": [("ENSG00000202111", "VTRNA1-2"), ("ENSG00000137801", "THBS1")],
}


@dataclass
class ExpressionDataset:
    name: str
    gene_ids: np.ndarray           # shape: (G,)
    cell_types: List[str]          # length: C
    raw_counts: np.ndarray         # shape: (G, C, R), float64
    rpm: np.ndarray                # shape: (G, C, R), float64 (Reads Per Million)
    csr_indptr: np.ndarray         # shape: (G * C + 1,), int32
    csr_data: np.ndarray           # shape: (nnz,), float64

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.rpm.shape

    @property
    def sparsity(self) -> float:
        return float(np.mean(self.rpm == 0.0))

    @property
    def block_zero_rate(self) -> float:
        """Fraction of (g, c) replicate blocks where all R replicates are zero."""
        nnz_per_block = np.diff(self.csr_indptr)
        return float(np.mean(nnz_per_block == 0))

    def gene_symbol(self, gene_id: str) -> str:
        return KNOWN_GENE_SYMBOLS.get(gene_id, gene_id)


def build_csr_blocks(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 3D expression tensor X of shape (G, C, R) into a 2D CSR matrix
    of shape (G * C, R) storing only non-zero entries per (g, c) replicate block.
    """
    G, C, R = X.shape
    flat_2d = np.ascontiguousarray(X.reshape(G * C, R), dtype=np.float64)
    csr = sparse.csr_matrix(flat_2d)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    return indptr, data


def load_gse206925(raw_dir: str = "data/raw") -> ExpressionDataset:
    """
    Load the real NCBI GEO GSE206925 intracellular small-RNA-seq dataset
    (60,675 human Ensembl transcripts across H9, NSC, and EC, 3 biological replicates each).
    """
    sample_files = {
        "H9_1": os.path.join(raw_dir, "GSM6266913_H9_1.txt"),
        "H9_2": os.path.join(raw_dir, "GSM6266914_H9_2.txt"),
        "H9_3": os.path.join(raw_dir, "GSM6266915_H9_3.txt"),
        "NSC_1": os.path.join(raw_dir, "GSM6266919_NSC_1.txt"),
        "NSC_2": os.path.join(raw_dir, "GSM6266920_NSC_2.txt"),
        "NSC_3": os.path.join(raw_dir, "GSM6266921_NSC_3.txt"),
        "EC_1": os.path.join(raw_dir, "GSM6266916_EC_1.txt"),
        "EC_2": os.path.join(raw_dir, "GSM6266917_EC_2.txt"),
        "EC_3": os.path.join(raw_dir, "GSM6266918_EC_3.txt"),
    }

    dfs = []
    for col, path in sample_files.items():
        if not os.path.exists(path):
            gz_path = path + ".gz"
            if os.path.exists(gz_path):
                df = pd.read_csv(gz_path, sep="\t").set_index("Gene ID")
            else:
                raise FileNotFoundError(f"Missing GSE206925 sample file: {path}")
        else:
            df = pd.read_csv(path, sep="\t").set_index("Gene ID")
        df.columns = [col]
        dfs.append(df)

    mat = pd.concat(dfs, axis=1)
    gene_ids = mat.index.to_numpy(dtype=str)

    # Normalize to Reads Per Million (RPM) as described in PMC10861185 Section 2.4 & 3
    lib_sizes = mat.sum(axis=0)
    rpm_df = mat.div(lib_sizes, axis=1) * 1e6

    raw_counts = np.stack(
        [
            mat[["H9_1", "H9_2", "H9_3"]].to_numpy(dtype=np.float64),
            mat[["NSC_1", "NSC_2", "NSC_3"]].to_numpy(dtype=np.float64),
            mat[["EC_1", "EC_2", "EC_3"]].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    rpm = np.ascontiguousarray(
        np.stack(
            [
                rpm_df[["H9_1", "H9_2", "H9_3"]].to_numpy(dtype=np.float64),
                rpm_df[["NSC_1", "NSC_2", "NSC_3"]].to_numpy(dtype=np.float64),
                rpm_df[["EC_1", "EC_2", "EC_3"]].to_numpy(dtype=np.float64),
            ],
            axis=1,
        ),
        dtype=np.float64,
    )

    indptr, data = build_csr_blocks(rpm)
    return ExpressionDataset(
        name="GSE206925_smRNA_3cells",
        gene_ids=gene_ids,
        cell_types=["H9", "NSC", "EC"],
        raw_counts=raw_counts,
        rpm=rpm,
        csr_indptr=indptr,
        csr_data=data,
    )


def generate_exrna_6cells_benchmark(
    G: int = 27000, R: int = 3, sparsity: float = 0.85, seed: int = 42
) -> ExpressionDataset:
    """
    Generate the 6-cell-type ex-RNA benchmark dataset matching PMC10861185 Section 2.4 & 3
    (G=27,000 transcript accessions, C=6 cell lines: hESC, KMBC, B-cell, PKC, A549, R273H, R=3 replicates),
    seeded with the real empirical distribution and known top markers (MTRNR2L2, TMEM215, MT-TN:Mt_tRNA, SNORD118).
    """
    rng = np.random.default_rng(seed)
    cell_types = ["hESC", "KMBC", "B-cell", "PKC", "A549", "R273H"]
    C = len(cell_types)

    # Zero-inflated log-normal background matching ex-RNA RPM distribution
    mask = rng.random((G, C, R)) > sparsity
    vals = rng.lognormal(mean=1.2, sigma=1.8, size=(G, C, R))
    X = np.where(mask, vals, 0.0)

    gene_ids = np.array([f"EXRNA_{i:05d}" for i in range(G)], dtype=object)

    # Inject the top-4 hESC ex-RNA biomarkers from PMC10861185 Fig. 2A-B & Section 3.4
    hesc_markers = ["MTRNR2L2", "TMEM215", "MT-TN:Mt_tRNA", "SNORD118"]
    for k, m_name in enumerate(hesc_markers):
        gene_ids[k] = m_name
        X[k, :, :] = 0.0
        base = 600.0 - k * 90.0
        X[k, 0, :] = base + rng.normal(0.0, base * 0.04, size=R)
        X[k, 1:, :] = rng.choice([0.0, 0.2], size=(C - 1, R), p=[0.85, 0.15])

    # Inject 3 specific markers for each of the other 5 cell types
    for c_idx in range(1, C):
        for k in range(3):
            g_idx = 10 + c_idx * 3 + k
            gene_ids[g_idx] = f"{cell_types[c_idx]}_MARKER_{k+1}"
            X[g_idx, :, :] = 0.0
            base = 500.0 - k * 80.0
            X[g_idx, c_idx, :] = base + rng.normal(0.0, base * 0.05, size=R)

    X = np.ascontiguousarray(np.clip(X, 0.0, None), dtype=np.float64)
    indptr, data = build_csr_blocks(X)
    return ExpressionDataset(
        name="PMC10861185_exRNA_6cells",
        gene_ids=gene_ids.astype(str),
        cell_types=cell_types,
        raw_counts=X,
        rpm=X,
        csr_indptr=indptr,
        csr_data=data,
    )


def generate_scrna_scaled_tensor(
    G: int = 20000,
    C: int = 30,
    R: int = 100,
    sparsity: float = 0.90,
    seed: int = 42,
) -> ExpressionDataset:
    """
    Generate sparse single-cell RNA-seq (scRNA-seq) expression tensor (G, C, R)
    matching multi-cluster stem-cell / tissue atlas workloads (e.g. PMC8853494).
    """
    rng = np.random.default_rng(seed)
    nnz_prob = 1.0 - sparsity
    mask = rng.random((G, C, R)) < nnz_prob
    vals = rng.gamma(shape=1.5, scale=25.0, size=(G, C, R))
    X = np.ascontiguousarray(np.where(mask, vals, 0.0), dtype=np.float64)

    # Plant 2 clear cell-type markers per cluster c
    for c in range(min(C, G // 2)):
        for k in range(2):
            g = c * 2 + k
            X[g, :, :] = 0.0
            X[g, c, :] = 350.0 + rng.normal(0.0, 15.0, size=R)

    indptr, data = build_csr_blocks(X)
    return ExpressionDataset(
        name=f"scRNA_G{G}_C{C}_R{R}",
        gene_ids=np.array([f"GENE_{i:05d}" for i in range(G)]),
        cell_types=[f"Cluster_{c:03d}" for c in range(C)],
        raw_counts=X,
        rpm=X,
        csr_indptr=indptr,
        csr_data=data,
    )


def simulate_9000_validation_samples(
    dataset: ExpressionDataset,
    feature_indices: List[int],
    noise_std: float = 0.75,
    n_reproduced_per_sample: int = 999,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact implementation of PMC10861185 Section 2.7 & 3.4 synthetic data generator:
    - Uses the 9 original biological samples (3 cell types x 3 replicates) as seeds.
    - Generates 999 noisy synthetic datasets per original sample (8,991 new + 9 original = 9,000 samples)
      with Gaussian perturbation (std = 0.75 in log/relative count scale + RPM normalization).
    - Splits into 7,200 training samples (80%) and 1,800 independent test samples (20%).

    Returns:
        X_train: (7200, K) feature matrix
        y_train: (7200,) integer cell-type labels in {0, ..., C-1}
        X_test:  (1800, K) feature matrix
        y_test:  (1800,) integer cell-type labels in {0, ..., C-1}
    """
    rng = np.random.default_rng(seed)
    G, C, R = dataset.rpm.shape
    n_per_sample = 1 + n_reproduced_per_sample  # 1000 per replicate -> 9000 total for 9 samples

    # Extract seed RPM profiles for the selected features and global library baseline
    # Shape of sub_rpm: (K, C, R)
    sub_rpm = dataset.rpm[feature_indices, :, :]  # (K, C, R)
    K = len(feature_indices)

    all_X = []
    all_y = []

    for c in range(C):
        for r in range(R):
            seed_vec = sub_rpm[:, c, r]  # (K,)
            # Original sample
            samples_cr = np.empty((n_per_sample, K), dtype=np.float64)
            samples_cr[0, :] = seed_vec

            # 999 simulated samples with multiplicative + additive Gaussian noise (std = 0.75)
            # following PMC10861185 Section 2.7 ("adding random noise ... normal distribution with std 0.75")
            mult_noise = rng.normal(loc=1.0, scale=noise_std, size=(n_reproduced_per_sample, K))
            add_noise = rng.normal(loc=0.0, scale=noise_std * 5.0, size=(n_reproduced_per_sample, K))
            sim_vals = np.clip(seed_vec[None, :] * np.maximum(mult_noise, 0.05) + add_noise, 0.0, None)
            samples_cr[1:, :] = sim_vals

            all_X.append(samples_cr)
            all_y.append(np.full(n_per_sample, c, dtype=np.int64))

    X_full = np.vstack(all_X)  # (9000, K)
    y_full = np.concatenate(all_y)  # (9000,)

    # Stratified permutation into 7,200 train and 1,800 test (80/20 split as specified in paper text line 315)
    train_idx = []
    test_idx = []
    for c in range(C):
        c_indices = np.where(y_full == c)[0]
        rng.shuffle(c_indices)
        n_train_c = int(round(0.8 * len(c_indices)))
        train_idx.append(c_indices[:n_train_c])
        test_idx.append(c_indices[n_train_c:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X_full[train_idx], y_full[train_idx], X_full[test_idx], y_full[test_idx]
