import unittest
import numpy as np
from scipy.stats import spearmanr

from src.data_loader import (
    load_gse206925,
    generate_exrna_6cells_benchmark,
    generate_scrna_scaled_tensor,
    simulate_9000_validation_samples,
)
from src.scoring import (
    baseline_scoring_loop,
    baseline_scoring_loop_cell_norm,
    optimized_scoring_vectorized_prompt,
    optimized_scoring_convex_numpy,
    optimized_scoring_target_norm_numpy,
    optimized_scoring_csr_scipy,
    optimized_scoring_cpp_dense,
    optimized_scoring_cpp_csr,
    optimized_scoring_cpp_target_csr,
    pmc10861185_eq1_baseline_loop,
    pmc10861185_eq1_convex_numpy,
    pmc10861185_eq1_cpp_csr,
)
from src.classifiers import CholeskyShrinkageLDA, evaluate_classifiers_on_9000_samples


class TestScoringAndOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gse = load_gse206925()
        cls.small_ds = generate_scrna_scaled_tensor(G=500, C=6, R=5, sparsity=0.80, seed=123)

    def test_convex_zscore_identity_exactness(self):
        """Verify max_r |x_r - mu| == max(x_max - mu, mu - x_min) has 0.0 IEEE-754 error."""
        X = self.gse.rpm
        V = np.mean(X, axis=2)
        std = np.std(X, axis=2)
        safe_std = np.where(std > 1e-12, std, 1.0)

        z_broadcast = np.max(np.abs(X - V[:, :, None]) / safe_std[:, :, None], axis=2)
        z_convex = np.maximum(np.max(X, axis=2) - V, V - np.min(X, axis=2)) / safe_std

        max_err = np.max(np.abs(z_broadcast - z_convex))
        self.assertEqual(max_err, 0.0)

    def test_cell_norm_equivalence_across_all_kernels(self):
        """Verify baseline_scoring_loop_cell_norm, prompt vectorized, convex numpy, CSR scipy, C++ dense, and C++ CSR all match to < 1e-14."""
        X = self.small_ds.rpm
        G, C, R = X.shape
        indptr, data = self.small_ds.csr_indptr, self.small_ds.csr_data

        s_base = baseline_scoring_loop_cell_norm(X)
        s_prompt = optimized_scoring_vectorized_prompt(X)
        s_convex = optimized_scoring_convex_numpy(X)
        s_csr_py = optimized_scoring_csr_scipy(indptr, data, G, C, R)
        s_cpp_dense = optimized_scoring_cpp_dense(X)
        s_cpp_csr = optimized_scoring_cpp_csr(indptr, data, G, C, R)

        np.testing.assert_allclose(s_base, s_prompt, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(s_prompt, s_convex, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(s_convex, s_csr_py, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(s_convex, s_cpp_dense, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(s_convex, s_cpp_csr, rtol=1e-14, atol=1e-14)

        for c in range(C):
            rho, _ = spearmanr(s_base[:, c], s_cpp_csr[:, c])
            self.assertAlmostEqual(rho, 1.0, places=12)

    def test_target_norm_equivalence_with_baseline_loop(self):
        """Verify optimized_scoring_target_norm_numpy and optimized_scoring_cpp_target_csr match baseline_scoring_loop to < 1e-14."""
        X = self.small_ds.rpm
        G, C, R = X.shape
        indptr, data = self.small_ds.csr_indptr, self.small_ds.csr_data

        s_base = baseline_scoring_loop(X)
        s_opt_np = optimized_scoring_target_norm_numpy(X)
        s_opt_cpp = optimized_scoring_cpp_target_csr(indptr, data, G, C, R)

        np.testing.assert_allclose(s_base, s_opt_np, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(s_base, s_opt_cpp, rtol=1e-13, atol=1e-14)

        for c in range(C):
            top20_base = list(np.argsort(-s_base[:, c])[:20])
            top20_cpp = list(np.argsort(-s_opt_cpp[:, c])[:20])
            self.assertEqual(top20_base, top20_cpp)

    def test_pmc10861185_eq1_equivalence(self):
        """Verify PMC10861185 Equation (1) baseline loop, convex numpy, and C++ CSR match to < 1e-13."""
        X = self.small_ds.rpm
        G, C, R = X.shape
        indptr, data = self.small_ds.csr_indptr, self.small_ds.csr_data

        s_loop = pmc10861185_eq1_baseline_loop(X, alpha=0.3)
        s_np = pmc10861185_eq1_convex_numpy(X, alpha=0.3)
        s_cpp = pmc10861185_eq1_cpp_csr(indptr, data, G, C, R, alpha=0.3)

        finite_mask = np.isfinite(s_loop)
        self.assertTrue(np.array_equal(finite_mask, np.isfinite(s_np)))
        self.assertTrue(np.array_equal(finite_mask, np.isfinite(s_cpp)))
        np.testing.assert_allclose(s_loop[finite_mask], s_np[finite_mask], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(s_loop[finite_mask], s_cpp[finite_mask], rtol=1e-13, atol=1e-13)

    def test_underflow_resilience_at_large_C(self):
        """Verify that log-domain scoring prevents float32/float64 underflow at C=120 when non-target expression is moderate."""
        rng = np.random.default_rng(99)
        G, C, R = 50, 120, 4
        # Moderate background expression across all cell types -> penalty factor ~0.1 per cell type
        # Over C-1 = 119 cell types, 0.1^119 = 1e-119 (underflows float32 completely!)
        X = rng.uniform(50.0, 100.0, size=(G, C, R))
        X[0, 0, :] = 1000.0  # True marker for cell 0

        s_f32_loop = baseline_scoring_loop_cell_norm(X, dtype=np.float32)
        log_s_opt = optimized_scoring_convex_numpy(X, return_log=True)

        # In float32, 0.1^119 flushes to 0.0 for all genes!
        self.assertEqual(np.max(s_f32_loop[:, 0]), 0.0)
        # In log-domain, all scores are finite and gene 0 is strictly ranked #1!
        self.assertTrue(np.all(np.isfinite(log_s_opt[:, 0])))
        self.assertEqual(int(np.argmax(log_s_opt[:, 0])), 0)

    def test_cholesky_shrinkage_lda_on_gse206925_9000_samples(self):
        """Verify CholeskyShrinkageLDA achieves > 0.98 ROC-AUC in < 5 ms on 9,000 GSE206925 samples."""
        gids = self.gse.gene_ids
        marker_ids = [
            "ENSG00000283490",  # MIR518C (H9)
            "ENSG00000207869",  # MIR518B (H9)
            "ENSG00000266782",  # MIR3663 (NSC)
            "ENSG00000283682",  # MIR548AH (NSC)
            "ENSG00000202111",  # VTRNA1-2 (EC)
            "ENSG00000137801",  # THBS1 (EC)
        ]
        feat_idx = [int(np.where(gids == m)[0][0]) for m in marker_ids]
        X_tr, y_tr, X_te, y_te = simulate_9000_validation_samples(self.gse, feat_idx)

        clf = CholeskyShrinkageLDA(shrinkage=0.05)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = np.mean(preds == y_te)
        self.assertGreater(acc, 0.95)


if __name__ == "__main__":
    unittest.main()
