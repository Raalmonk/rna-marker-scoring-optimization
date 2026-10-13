#include <cmath>
#include <cstdint>
#include <algorithm>
#include <dispatch/dispatch.h>

extern "C" {

// ============================================================================
// 1. Fused Dense Log-Domain Scoring Kernel (Cell-Normalized)
//    Complexity: O(G * C * R) time, L1 register-resident per (g, c) block!
//    Uses Convex Extreme-Point Z-score identity:
//      max_r |x_r - mean| == max(x_max - mean, mean - x_min)
// ============================================================================
void fused_dense_scoring_cell_norm(
    const double* __restrict__ X,       // shape: (G, C, R) contiguous
    double* __restrict__ out_scores,    // shape: (G, C)
    int64_t G,
    int64_t C,
    int64_t R,
    double eps,
    int return_log
) {
    double* V_buf = new double[G * C];
    double* Z_buf = new double[G * C];
    double* V_max = new double[C];

    const double inv_R = 1.0 / static_cast<double>(R);

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        const double* X_g = X + g * C * R;
        double* V_g = V_buf + g * C;
        double* Z_g = Z_buf + g * C;

        for (int64_t c = 0; c < C; ++c) {
            const double* rep = X_g + c * R;
            double sum = 0.0;
            double x_min = rep[0];
            double x_max = rep[0];

            for (int64_t r = 0; r < R; ++r) {
                double val = rep[r];
                sum += val;
                if (val < x_min) x_min = val;
                if (val > x_max) x_max = val;
            }

            double mean = sum * inv_R;
            V_g[c] = mean;

            if (x_max == x_min || mean <= eps) {
                Z_g[c] = (x_max == x_min) ? -1.0 : 0.0;
                continue;
            }

            double sum_sq_diff = 0.0;
            for (int64_t r = 0; r < R; ++r) {
                double diff = rep[r] - mean;
                sum_sq_diff += diff * diff;
            }
            double std_val = std::sqrt(sum_sq_diff * inv_R);

            if (std_val > eps) {
                double max_dev = std::max(x_max - mean, mean - x_min);
                Z_g[c] = max_dev / std_val;
            } else {
                Z_g[c] = -1.0;
            }
        }
    });

    dispatch_apply(C, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        double max_v = eps;
        for (int64_t g = 0; g < G; ++g) {
            double v = V_buf[g * C + c];
            if (v > max_v) max_v = v;
        }
        V_max[c] = max_v;
    });

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        const double* V_g = V_buf + g * C;
        const double* Z_g = Z_buf + g * C;
        double* out_g = out_scores + g * C;

        double row_sum_penalty = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double v_norm = V_g[c] / V_max[c];
            double z_non_target = (Z_g[c] > 0.0) ? Z_g[c] : 0.0;
            double pen = -std::log1p(v_norm * (1.0 + z_non_target));
            out_g[c] = pen;
            row_sum_penalty += pen;
        }

        for (int64_t t = 0; t < C; ++t) {
            double v_t = V_g[t];
            if (v_t <= eps) {
                out_g[t] = return_log ? -INFINITY : 0.0;
            } else {
                double v_norm_t = v_t / V_max[t];
                double z_target = (Z_g[t] >= 0.0) ? Z_g[t] : 1.0;
                double log_target = std::log(std::max(v_norm_t, eps)) - std::log1p(z_target);
                double log_non_target = row_sum_penalty - out_g[t];
                double log_score = log_target + log_non_target;
                out_g[t] = return_log ? log_score : std::exp(log_score);
            }
        }
    });

    delete[] V_buf;
    delete[] Z_buf;
    delete[] V_max;
}

// ============================================================================
// 2. Fused CSR Sparse Zero-Skipping Log-Domain Scoring Kernel
//    Complexity: O(nnz(X) + G * C) with exact centered zero-skipping variance
//    sum_{r=1}^R (x_r - mu)^2 = sum_{x_r != 0} (x_r - mu)^2 + (R - nnz) * mu^2
// ============================================================================
void fused_csr_scoring_cell_norm(
    const int32_t* __restrict__ indptr, // length: G * C + 1
    const double* __restrict__ data,    // length: nnz
    double* __restrict__ out_scores,    // shape: (G, C)
    int64_t G,
    int64_t C,
    int64_t R,
    double eps,
    int return_log
) {
    double* V_buf = new double[G * C];
    double* Z_buf = new double[G * C];
    double* V_max = new double[C];

    const double inv_R = 1.0 / static_cast<double>(R);

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        double* V_g = V_buf + g * C;
        double* Z_g = Z_buf + g * C;
        int64_t row_base = g * C;

        for (int64_t c = 0; c < C; ++c) {
            int32_t start = indptr[row_base + c];
            int32_t end = indptr[row_base + c + 1];
            int32_t nnz = end - start;

            if (nnz == 0) {
                V_g[c] = 0.0;
                Z_g[c] = -1.0;
                continue;
            }

            double sum = 0.0;
            double x_min_nz = data[start];
            double x_max_nz = data[start];

            for (int32_t idx = start; idx < end; ++idx) {
                double val = data[idx];
                sum += val;
                if (val < x_min_nz) x_min_nz = val;
                if (val > x_max_nz) x_max_nz = val;
            }

            double mean = sum * inv_R;
            V_g[c] = mean;

            double sum_sq_diff = static_cast<double>(R - nnz) * mean * mean;
            for (int32_t idx = start; idx < end; ++idx) {
                double diff = data[idx] - mean;
                sum_sq_diff += diff * diff;
            }
            double std_val = std::sqrt(sum_sq_diff * inv_R);

            if (std_val > eps && mean > eps) {
                double x_min = (nnz < R) ? 0.0 : x_min_nz;
                double max_dev = std::max(x_max_nz - mean, mean - x_min);
                Z_g[c] = max_dev / std_val;
            } else {
                Z_g[c] = (std_val <= eps) ? -1.0 : 0.0;
            }
        }
    });

    dispatch_apply(C, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        double max_v = eps;
        for (int64_t g = 0; g < G; ++g) {
            double v = V_buf[g * C + c];
            if (v > max_v) max_v = v;
        }
        V_max[c] = max_v;
    });

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        const double* V_g = V_buf + g * C;
        const double* Z_g = Z_buf + g * C;
        double* out_g = out_scores + g * C;

        double row_sum_penalty = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double v_t = V_g[c];
            double pen = 0.0;
            if (v_t > eps) {
                double v_norm = v_t / V_max[c];
                double z_non_target = (Z_g[c] > 0.0) ? Z_g[c] : 0.0;
                pen = -std::log1p(v_norm * (1.0 + z_non_target));
            }
            out_g[c] = pen;
            row_sum_penalty += pen;
        }

        for (int64_t t = 0; t < C; ++t) {
            double v_t = V_g[t];
            if (v_t <= eps) {
                out_g[t] = return_log ? -INFINITY : 0.0;
            } else {
                double v_norm_t = v_t / V_max[t];
                double z_target = (Z_g[t] >= 0.0) ? Z_g[t] : 1.0;
                double log_target = std::log(std::max(v_norm_t, eps)) - std::log1p(z_target);
                double log_non_target = row_sum_penalty - out_g[t];
                double log_score = log_target + log_non_target;
                out_g[t] = return_log ? log_score : std::exp(log_score);
            }
        }
    });

    delete[] V_buf;
    delete[] Z_buf;
    delete[] V_max;
}

// ============================================================================
// 3. Fused PMC10861185 Exact Equation (1) Additive Kernel (CSR Zero-Skipping)
// ============================================================================
void fused_pmc10861185_eq1_csr(
    const int32_t* __restrict__ indptr,
    const double* __restrict__ data,
    double* __restrict__ out_scores,
    int64_t G,
    int64_t C,
    int64_t R,
    double alpha,
    double eps
) {
    double* V_buf = new double[G * C];
    double* S_buf = new double[G * C];
    double* Z_buf = new double[G * C];
    double* V_max = new double[C];
    double* Z_max = new double[C];

    const double inv_R = 1.0 / static_cast<double>(R);

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        double* V_g = V_buf + g * C;
        double* S_g = S_buf + g * C;
        double* Z_g = Z_buf + g * C;
        int64_t row_base = g * C;

        for (int64_t c = 0; c < C; ++c) {
            int32_t start = indptr[row_base + c];
            int32_t end = indptr[row_base + c + 1];
            int32_t nnz = end - start;

            if (nnz == 0) {
                V_g[c] = 0.0;
                S_g[c] = 0.0;
                Z_g[c] = 0.0;
                continue;
            }

            double sum = 0.0;
            double x_min_nz = data[start];
            double x_max_nz = data[start];

            for (int32_t idx = start; idx < end; ++idx) {
                double val = data[idx];
                sum += val;
                if (val < x_min_nz) x_min_nz = val;
                if (val > x_max_nz) x_max_nz = val;
            }

            double mean = sum * inv_R;
            double sum_sq_diff = static_cast<double>(R - nnz) * mean * mean;
            for (int32_t idx = start; idx < end; ++idx) {
                double diff = data[idx] - mean;
                sum_sq_diff += diff * diff;
            }
            double std_val = std::sqrt(sum_sq_diff * inv_R);

            V_g[c] = mean;
            S_g[c] = std_val;
            if (std_val > eps && mean > eps) {
                double x_min = (nnz < R) ? 0.0 : x_min_nz;
                double max_dev = std::max(x_max_nz - mean, mean - x_min);
                Z_g[c] = max_dev / std_val;
            } else {
                Z_g[c] = 0.0;
            }
        }
    });

    dispatch_apply(C, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        double mv = eps;
        double mz = eps;
        for (int64_t g = 0; g < G; ++g) {
            double v = V_buf[g * C + c];
            double z = Z_buf[g * C + c];
            if (v > mv) mv = v;
            if (z > mz) mz = z;
        }
        V_max[c] = mv;
        Z_max[c] = mz;
    });

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        const double* V_g = V_buf + g * C;
        const double* S_g = S_buf + g * C;
        const double* Z_g = Z_buf + g * C;
        double* out_g = out_scores + g * C;

        double row_sum_nv = 0.0;
        double row_sum_nz = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double nv = V_g[c] / V_max[c];
            double nz = Z_g[c] / Z_max[c];
            out_g[c] = nv + nz;
            row_sum_nv += nv;
            row_sum_nz += nz;
        }
        double total_non_target_all = row_sum_nv + row_sum_nz;

        for (int64_t t = 0; t < C; ++t) {
            double v_t = V_g[t];
            if (v_t <= eps) {
                out_g[t] = INFINITY;
            } else {
                double target_term = (V_max[t] / v_t) + alpha * (S_g[t] / v_t);
                double non_target_term = total_non_target_all - out_g[t];
                out_g[t] = target_term + non_target_term;
            }
        }
    });

    delete[] V_buf;
    delete[] S_buf;
    delete[] Z_buf;
    delete[] V_max;
    delete[] Z_max;
}

// ============================================================================
// 4. Fused CSR Sparse Zero-Skipping Kernel (Target-Normalized: v_i / V_max[t])
// ============================================================================
void fused_csr_scoring_target_norm(
    const int32_t* __restrict__ indptr,
    const double* __restrict__ data,
    double* __restrict__ out_scores,
    int64_t G,
    int64_t C,
    int64_t R,
    double eps,
    int return_log
) {
    double* V_buf = new double[G * C];
    double* Z_buf = new double[G * C];
    double* W_buf = new double[G * C];
    double* V_max = new double[C];
    double* inv_V_max = new double[C];

    const double inv_R = 1.0 / static_cast<double>(R);

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        double* V_g = V_buf + g * C;
        double* Z_g = Z_buf + g * C;
        double* W_g = W_buf + g * C;
        int64_t row_base = g * C;

        for (int64_t c = 0; c < C; ++c) {
            int32_t start = indptr[row_base + c];
            int32_t end = indptr[row_base + c + 1];
            int32_t nnz = end - start;

            if (nnz == 0) {
                V_g[c] = 0.0;
                Z_g[c] = -1.0;
                W_g[c] = 0.0;
                continue;
            }

            double sum = 0.0;
            double x_min_nz = data[start];
            double x_max_nz = data[start];

            for (int32_t idx = start; idx < end; ++idx) {
                double val = data[idx];
                sum += val;
                if (val < x_min_nz) x_min_nz = val;
                if (val > x_max_nz) x_max_nz = val;
            }

            double mean = sum * inv_R;
            double sum_sq_diff = static_cast<double>(R - nnz) * mean * mean;
            for (int32_t idx = start; idx < end; ++idx) {
                double diff = data[idx] - mean;
                sum_sq_diff += diff * diff;
            }
            double std_val = std::sqrt(sum_sq_diff * inv_R);

            V_g[c] = mean;
            if (std_val > eps && mean > eps) {
                double x_min = (nnz < R) ? 0.0 : x_min_nz;
                double max_dev = std::max(x_max_nz - mean, mean - x_min);
                double z = max_dev / std_val;
                Z_g[c] = z;
                W_g[c] = mean * (1.0 + z);
            } else {
                Z_g[c] = (std_val <= eps) ? -1.0 : 0.0;
                W_g[c] = mean;
            }
        }
    });

    dispatch_apply(C, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
        double max_v = 0.0;
        for (int64_t g = 0; g < G; ++g) {
            double v = V_buf[g * C + c];
            if (v > max_v) max_v = v;
        }
        V_max[c] = (max_v > 0.0) ? max_v : 1.0;
        inv_V_max[c] = 1.0 / V_max[c];
    });

    dispatch_apply(G, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t g) {
        const double* V_g = V_buf + g * C;
        const double* Z_g = Z_buf + g * C;
        const double* W_g = W_buf + g * C;
        double* out_g = out_scores + g * C;

        int64_t active_c[512];
        int64_t n_active = 0;
        if (C <= 512) {
            for (int64_t c = 0; c < C; ++c) {
                if (W_g[c] > 0.0) {
                    active_c[n_active++] = c;
                }
            }
        }

        for (int64_t t = 0; t < C; ++t) {
            double v_t = V_g[t];
            if (v_t <= eps) {
                out_g[t] = return_log ? -INFINITY : 0.0;
                continue;
            }
            double inv_vm = inv_V_max[t];
            double z_target = (Z_g[t] >= 0.0) ? Z_g[t] : 1.0;
            double log_target = std::log(v_t * inv_vm) - std::log1p(z_target);

            double log_non_target = 0.0;
            if (C <= 512) {
                for (int64_t k = 0; k < n_active; ++k) {
                    int64_t i = active_c[k];
                    if (i != t) {
                        log_non_target -= std::log1p(W_g[i] * inv_vm);
                    }
                }
            } else {
                for (int64_t i = 0; i < C; ++i) {
                    if (i != t && W_g[i] > 0.0) {
                        log_non_target -= std::log1p(W_g[i] * inv_vm);
                    }
                }
            }
            double log_score = log_target + log_non_target;
            out_g[t] = return_log ? log_score : std::exp(log_score);
        }
    });

    delete[] V_buf;
    delete[] Z_buf;
    delete[] W_buf;
    delete[] V_max;
    delete[] inv_V_max;
}

} // extern "C"
