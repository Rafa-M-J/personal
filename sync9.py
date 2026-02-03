from __future__ import annotations



from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import argparse
import csv
import gc
import json
import base64
import os
import time
import threading
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Optional: PyTorch on-the-fly DataLoader integration (used for multi-core batch generation)
try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader, get_worker_info
except Exception:  # pragma: no cover
    torch = None
    IterableDataset = object  # type: ignore
    DataLoader = None  # type: ignore
    get_worker_info = None  # type: ignore

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

import psutil


# Sync version
SYNC_VERSION = "sync9"


# ---------------------------
# 1) Activation / Input prior
# ---------------------------

def _activation(name: str):
    """Return an activation function (callable) given its string name."""
    name = name.lower()
    if name == "tanh":
        return np.tanh
    if name in ("identity", "linear", "none"):
        return lambda x: x
    if name in ("leakyrelu", "leaky_relu"):
        return lambda x: np.where(x >= 0, x, 0.01 * x)
    if name == "elu":
        # NOTE: np.where evaluates both branches; avoid overflow by computing expm1 only on the negative part.
        return lambda x: np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
    raise ValueError(f"Unknown activation: {name}")


def _sample_mixed_inputs(
    rng: np.random.Generator,
    n: int,
    d: int,
    dtype=np.float32,
) -> np.ndarray:
    """Sample exogenous/root variables from a per-dimension mixture distribution."""
    dist_type = rng.integers(0, 4, size=d)  # 0 Normal, 1 Laplace, 2 Student-t, 3 Uniform
    Z0 = np.empty((n, d), dtype=np.float32)
    for j, t in enumerate(dist_type):
        if t == 0:
            Z0[:, j] = rng.normal(0.0, 1.0, size=n)
        elif t == 1:
            Z0[:, j] = rng.laplace(0.0, 1.0, size=n)
        elif t == 2:
            df = float(rng.integers(3, 10))  # df>=3 => finite variance (more stable for large-scale generation)
            Z0[:, j] = rng.standard_t(df=df, size=n)
        else:
            Z0[:, j] = rng.uniform(-2.0, 2.0, size=n)
    return Z0.astype(dtype, copy=False)


# ------------------------------------
# 2) Sparse DAG sampling (CSR weights)
# ------------------------------------

def build_sparse_csr(
    rng: np.random.Generator,
    out_dim: int,
    in_dim: int,
    fan_in: int,
    weight_std: float,
    dropout: float = 0.0,
    row_index_ranges: Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = None,
    dtype=np.float32,
) -> sp.csr_matrix:
    """Build a sparse weight matrix W in CSR format with shape (out_dim, in_dim)."""
    if fan_in < 1:
        raise ValueError("fan_in must be >= 1")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must be in [0,1).")

    if row_index_ranges is None:
        idx = rng.integers(0, in_dim, size=(out_dim, fan_in), dtype=np.int32)
    else:
        mask, r1, r2 = row_index_ranges
        low1, high1 = r1
        low2, high2 = r2
        idx = np.empty((out_dim, fan_in), dtype=np.int32)
        n1 = int(mask.sum())
        n2 = out_dim - n1
        if n1:
            idx[mask] = rng.integers(low1, high1, size=(n1, fan_in), dtype=np.int32)
        if n2:
            idx[~mask] = rng.integers(low2, high2, size=(n2, fan_in), dtype=np.int32)

    data = rng.normal(0.0, weight_std, size=(out_dim, fan_in)).astype(dtype, copy=False)

    if dropout > 0.0:
        keep = rng.random(size=(out_dim, fan_in)) >= dropout
        row_has = keep.any(axis=1)
        if not np.all(row_has):
            rows = np.where(~row_has)[0]
            cols = rng.integers(0, fan_in, size=len(rows))
            keep[rows, cols] = True

        nnz_per_row = keep.sum(axis=1).astype(np.int64)
        indptr = np.zeros(out_dim + 1, dtype=np.int64)
        np.cumsum(nnz_per_row, out=indptr[1:])
        flat_keep = keep.ravel()
        indices = idx.ravel()[flat_keep].astype(np.int32, copy=False)
        data_flat = data.ravel()[flat_keep].astype(dtype, copy=False)
        return sp.csr_matrix((data_flat, indices, indptr), shape=(out_dim, in_dim), dtype=dtype)

    indptr = np.arange(0, out_dim * fan_in + 1, fan_in, dtype=np.int64)
    return sp.csr_matrix((data.ravel(), idx.ravel(), indptr), shape=(out_dim, in_dim), dtype=dtype)


# -----------------------------------------
# 3) Sparse linear in chunks (threading)
# -----------------------------------------

def sparse_linear_in_chunks(
    W: sp.csr_matrix,
    H: np.ndarray,
    block_rows: int = 2048,
    n_jobs: int = 1,
    dtype=np.float32,
) -> np.ndarray:
    """Compute out = (W @ H.T).T by chunking rows of W (W is CSR, H is dense)."""
    if not sp.isspmatrix_csr(W):
        W = W.tocsr()
    N, in_dim = H.shape
    out_dim = W.shape[0]
    if W.shape[1] != in_dim:
        raise ValueError("Shape mismatch")

    out = np.empty((N, out_dim), dtype=dtype)

    def work(start: int, end: int):
        out[:, start:end] = W[start:end].dot(H.T).T

    if n_jobs <= 1 or out_dim <= block_rows:
        for start in range(0, out_dim, block_rows):
            end = min(out_dim, start + block_rows)
            work(start, end)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = []
            for start in range(0, out_dim, block_rows):
                end = min(out_dim, start + block_rows)
                futs.append(ex.submit(work, start, end))
            for f in as_completed(futs):
                f.result()

    return out


# -----------------------------------------
# 4) Utils: meta serialization + IO
# -----------------------------------------

def _to_jsonable(obj: Any) -> Any:
    """Convert common numpy / python objects into JSON-serializable types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # Handle dtypes / scalar types / paths for config snapshots.
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, type):
        return getattr(obj, '__name__', str(obj))
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_dataset_npz(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    meta: Dict[str, Any],
    compress: bool = True,
) -> None:
    """Save a dataset to NPZ (with JSON metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_json = json.dumps(_to_jsonable(meta), ensure_ascii=False)
    # Store meta_json as a normal unicode array (NOT object dtype) so we can load without allow_pickle.
    if compress:
        np.savez_compressed(path, X=X, y=y, y_hat=y_hat, meta_json=np.array(meta_json))
    else:
        np.savez(path, X=X, y=y, y_hat=y_hat, meta_json=np.array(meta_json))


def load_dataset_npz(path: str | Path) -> Dict[str, Any]:
    """Load a dataset from NPZ (metadata is decoded from JSON)."""
    path = Path(path)

    # Fast path: our current format stores meta_json as a normal unicode array,
    # so allow_pickle=False is safe.
    try:
        with np.load(path, allow_pickle=False) as z:
            X = z['X']
            y = z['y']
            y_hat = z['y_hat']
            meta_json = z['meta_json'].item()
    except ValueError:
        # Backward-compat: older files stored meta_json as object dtype (requires allow_pickle=True).
        with np.load(path, allow_pickle=True) as z:
            X = z['X']
            y = z['y']
            y_hat = z['y_hat']
            meta_json = z['meta_json'].item()

    if isinstance(meta_json, (bytes, np.bytes_)):
        meta_json = meta_json.decode('utf-8')
    else:
        meta_json = str(meta_json)
    meta = json.loads(meta_json)
    return {'X': X, 'y': y, 'y_hat': y_hat, 'meta': meta}


# ---------------------------------
# 5) Config and Generator
# ---------------------------------

@dataclass
class LargePSmallNSynthConfig:
    """All knobs that control the synthetic data generator."""

    # Target regime
    n_features: int = 10_000
    n_samples: int = 256


    # Within each generated dataset, we often treat the first n_train samples as "training/context"
    # samples (used for scaling statistics and other train-only transforms).
    # If n_train is None, we use train_fraction * n_samples (default train_fraction=1.0 => use all samples).
    n_train: Optional[int] = None
    train_fraction: float = 1.0
    # X generator switch
    x_generator: str = "scm"  # {"scm", "bnn", "gaussian"}

    # Hard constraint: number of label-relevant features must be <= max_relevant_features
    max_relevant_features: int = 100

    # Pool size for "signal" features (can be larger than max_relevant_features).
    # This allows experiments like: signal pool = 500, but true relevant cap = 100.
    max_signal_pool_features: int = 100

    # -------------------------
    # SCM architecture (layered sparse MLP treated as a DAG)
    # -------------------------
    latent_dim: int = 64
    hidden_dims: Tuple[int, ...] = (256,)
    fan_in_hidden: int = 8
    fan_in_features: int = 6
    dropout_hidden: float = 0.2
    dropout_features: float = 0.1
    weight_std: float = 1.0

    # Noise injected into SCM nodes
    noise_std_hidden: float = 0.3
    noise_std_features: float = 0.2

    # Activation
    activation_choices: Tuple[str, ...] = ("tanh", "leakyrelu", "elu", "identity")

    # Signal features (used as a pool for label-relevant features)
    signal_feature_fraction: float = 0.05
    label_latent_dim: int = 32

    # Engineering knobs
    feature_block_rows: int = 2048
    n_jobs_features: int = 1
    dtype: Any = np.float32

    # Feature scaling (applied column-wise to X after generation)
    # - "standard": (x - mean) / std
    # - "robust": (x - median) / (1.4826 * MAD)
    # - "none": no scaling
    x_scaling: str = "robust"  # {"standard", "robust", "none"}
    robust_mad_scale: float = 1.4826

    # Scaling of regression targets (y and y_hat) using training/context samples only.
    # - "none": no scaling
    # - "standard": (y - mean_train) / std_train
    # - "robust": (y - median_train) / (1.4826 * MAD_train)
    y_scaling: str = "robust"  # {"standard", "robust", "none"}
    y_robust_mad_scale: float = 1.4826

    # Numerical stability epsilon for scaling denominators (std, MAD, etc.)
    scale_eps: float = 1e-6

    # If True, add tiny jitter noise to near-constant columns (detected on the training subset)
    # to avoid NaNs / spurious "line" artifacts in correlation heatmaps.
    fix_near_constant_X: bool = True
    near_constant_jitter_std: float = 1e-3

    # -------------------------
    # BNN-overwrite generator
    # -------------------------
    bnn_base_generator: str = "gaussian"  # {"gaussian", "scm"}
    bnn_informative_features_range: Tuple[int, int] = (20, 100)  # m range (will be clamped to <= max_relevant_features)
    bnn_hidden_dims: Tuple[int, ...] = (256, 256)  # hidden widths for the random BNN
    bnn_weight_dropout: float = 0.1
    bnn_noise_std: float = 0.05
    bnn_input_scale_range: Tuple[float, float] = (0.5, 2.0)

    # -------------------------
    # Task / target controls
    # -------------------------
    # Default to regression since our end goal is sparse linear regression.
    task_type: str = "regression"  # {"classification", "regression"}
    y_generator: str = "linear_sparse"  # {"hidden", "linear_sparse", "rf_like", "gbdt_like"}

    # Multiclass label settings
    n_classes: Optional[int] = None
    min_class_count: int = 2

    # Generic eps for y = f(X) + eps
    snr_range: Tuple[float, float] = (0.5, 10.0)

    # SNR sampling mode
    # - "uniform": sample snr ~ Uniform(snr_range) (old behavior)
    # - "log_uniform": sample log(snr) ~ Uniform(log(lo), log(hi))
    # - "mixture": mixture of (moderate SNR) and (low SNR) with a fixed mixture weight (i.i.d.)
    snr_sampling: str = "mixture"  # {"uniform", "log_uniform", "mixture"}

    # Mixture/curriculum settings (used when snr_sampling == "mixture_curriculum")
    # We keep a moderate-SNR component for learnability, but gradually increase the probability of low-SNR tasks.
    snr_moderate_range: Tuple[float, float] = (1.0, 8.0)
    snr_low_range: Tuple[float, float] = (0.2, 2.0)
    snr_mixture_w_moderate_start: float = 0.8
    snr_mixture_w_moderate_end: float = 0.2
    snr_curriculum_steps: int = 50_000
    snr_curriculum_power: float = 1.0
    snr_mixture_log_uniform: bool = True
    eps_dist_choices: Tuple[str, ...] = ("normal", "laplace", "t", "lognormal")
    eps_df_range: Tuple[int, int] = (3, 10)

    # -------------------------
    # Sync9: Noise control via alpha-mixing (preferred, i.i.d.)
    # -------------------------
    # We generate y by first robust-standardizing the clean signal f(X) and raw noise eps,
    # then mixing:
    #   y_hat_used = sqrt(alpha) * f_tilde
    #   y         = y_hat_used + sqrt(1-alpha) * eps_tilde
    # With Var(f_tilde) ≈ Var(eps_tilde) ≈ 1, we get:
    #   R^2_target ≈ alpha,   SNR_target ≈ alpha/(1-alpha)
    #
    # This fixes sync8 issues where scaling only eps can yield generator-dependent y distributions,
    # and removes step-dependent curriculum drift (LLN independence concern).
    noise_control: str = "alpha_mix"  # {"alpha_mix", "snr_scale_eps"}  (snr_scale_eps = legacy)

    # alpha sampling
    alpha_sampling: str = "mixture"  # {"uniform", "beta", "mixture"}
    alpha_range: Tuple[float, float] = (0.02, 0.45)        # used by uniform/beta scaling
    alpha_beta_params: Tuple[float, float] = (2.0, 5.0)    # used when alpha_sampling=="beta" (sample u~Beta(a,b))
    alpha_low_range: Tuple[float, float] = (0.02, 0.30)    # mixture component (low R^2)
    alpha_moderate_range: Tuple[float, float] = (0.30, 0.45) # mixture component (moderate R^2)
    alpha_mixture_w_moderate: float = 0.10                 # P(component == moderate)

    # -------------------------
    # Sync9: Correlated Gaussian X (blockwise factor model)
    # -------------------------
    # Instead of iid N(0,1), we generate blockwise-correlated features:
    #   X_block = s * (U @ L^T) + sigma * E
    # where U ~ N(0,1)^{N x r}, L ~ N(0,1)^{b x r}, E ~ N(0,1)^{N x b}.
    gaussian_corr: str = "block_factor"  # {"block_factor", "independent"}
    gaussian_block_size_range: Tuple[int, int] = (25, 200)
    gaussian_factor_rank_range: Tuple[int, int] = (3, 20)
    gaussian_factor_strength_range: Tuple[float, float] = (0.7, 2.0)
    gaussian_idiosyncratic_noise_range: Tuple[float, float] = (0.3, 1.0)
    gaussian_shuffle_blocks: bool = True
    gaussian_shuffle_within_block: bool = True

    # -------------------------
    # Sync9: Feature ordering (trivial setting removal)
    # -------------------------
    # If True, apply a random column permutation per dataset and update all index-based metadata
    # (signal_idx, relevant_idx, etc.). This prevents "important features always come first".
    apply_feature_permutation: bool = True

    # Store the permutation in metadata (can be large when P is big). "auto" stores only for small P.
    store_feature_permutation: str = "auto"  # {"none", "auto", "list", "base64"}
    store_feature_permutation_max_p: int = 2000

    # Store dense relevant-feature mask in metadata (auto: only for small P to avoid huge JSON).
    store_relevant_mask: str = "auto"  # {"none", "auto", "dense"}
    store_relevant_mask_max_p: int = 2000

    # y = X beta + eps (sparse linear)
    beta_sparsity_range: Tuple[int, int] = (5, 50)
    beta_support_pool: str = "signal"  # {"signal", "all", "mixed"}
    beta_signal_fraction: float = 0.7
    beta_coef_dist: str = "normal"  # {"normal", "laplace", "t"}
    beta_coef_scale_range: Tuple[float, float] = (0.5, 2.0)

    # RF/GBDT-like teacher settings
    tree_feature_count_range: Tuple[int, int] = (20, 200)  # will be clamped to <= max_relevant_features
    tree_signal_fraction: float = 0.7
    tree_n_estimators_range: Tuple[int, int] = (20, 200)
    tree_max_depth_range: Tuple[int, int] = (2, 6)
    tree_min_leaf: int = 5
    tree_leaf_scale: float = 1.0
    gbdt_learning_rate_range: Tuple[float, float] = (0.05, 0.3)

    def __post_init__(self) -> None:
        """Clamp ranges to enforce the hard constraint globally."""
        # Basic sanity for dimensions
        self.n_features = int(self.n_features)
        self.n_samples = int(self.n_samples)
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")

        # Determine n_train (used for train-only scaling statistics)
        if self.n_train is None:
            tf = float(self.train_fraction)
            if not np.isfinite(tf):
                tf = 1.0
            tf = float(min(max(tf, 0.0), 1.0))
            n_tr = int(round(tf * float(self.n_samples)))
            n_tr = int(max(1, min(n_tr, self.n_samples)))
            self.n_train = n_tr
        else:
            self.n_train = int(max(1, min(int(self.n_train), int(self.n_samples))))

        # SNR curriculum sanity (clamp ranges & weights)
        self.snr_sampling = str(getattr(self, "snr_sampling", "uniform")).lower()
        lo, hi = float(self.snr_range[0]), float(self.snr_range[1])
        lo = max(lo, 1e-8)
        hi = max(hi, lo)
        self.snr_range = (lo, hi)

        mlo, mhi = float(self.snr_moderate_range[0]), float(self.snr_moderate_range[1])
        mlo = max(mlo, 1e-8)
        mhi = max(mhi, mlo)
        self.snr_moderate_range = (mlo, mhi)

        llo, lhi = float(self.snr_low_range[0]), float(self.snr_low_range[1])
        llo = max(llo, 1e-8)
        lhi = max(lhi, llo)
        self.snr_low_range = (llo, lhi)

        w0 = float(self.snr_mixture_w_moderate_start)
        w1 = float(self.snr_mixture_w_moderate_end)
        self.snr_mixture_w_moderate_start = float(min(max(w0, 0.0), 1.0))
        self.snr_mixture_w_moderate_end = float(min(max(w1, 0.0), 1.0))
        self.snr_curriculum_steps = int(max(1, int(self.snr_curriculum_steps)))
        self.snr_curriculum_power = float(max(1e-6, float(self.snr_curriculum_power)))

        # -------------------------
        # Sync9: alpha-mix noise control sanity
        # -------------------------
        self.noise_control = str(getattr(self, "noise_control", "alpha_mix")).lower()
        self.alpha_sampling = str(getattr(self, "alpha_sampling", "mixture")).lower()

        def _clamp_01_range(r: Tuple[float, float]) -> Tuple[float, float]:
            a0, a1 = float(r[0]), float(r[1])
            a0 = float(min(max(a0, 0.0), 1.0))
            a1 = float(min(max(a1, 0.0), 1.0))
            if a1 < a0:
                a0, a1 = a1, a0
            # avoid exact 0/1 to keep sqrt(1-alpha) stable
            a0 = float(max(a0, 1e-6))
            a1 = float(min(a1, 1.0 - 1e-6))
            return a0, a1

        self.alpha_range = _clamp_01_range(self.alpha_range)
        self.alpha_low_range = _clamp_01_range(self.alpha_low_range)
        self.alpha_moderate_range = _clamp_01_range(self.alpha_moderate_range)
        a_beta = getattr(self, "alpha_beta_params", (2.0, 5.0))
        self.alpha_beta_params = (float(max(1e-3, a_beta[0])), float(max(1e-3, a_beta[1])))
        self.alpha_mixture_w_moderate = float(min(max(float(getattr(self, "alpha_mixture_w_moderate", 0.1)), 0.0), 1.0))

        # -------------------------
        # Sync9: Gaussian correlation settings sanity
        # -------------------------
        self.gaussian_corr = str(getattr(self, "gaussian_corr", "block_factor")).lower()
        b0, b1 = int(self.gaussian_block_size_range[0]), int(self.gaussian_block_size_range[1])
        b0 = int(max(2, b0))
        b1 = int(max(b0, b1))
        self.gaussian_block_size_range = (b0, b1)

        r0, r1 = int(self.gaussian_factor_rank_range[0]), int(self.gaussian_factor_rank_range[1])
        r0 = int(max(1, r0))
        r1 = int(max(r0, r1))
        self.gaussian_factor_rank_range = (r0, r1)

        # factor strength / idiosyncratic noise
        s0, s1 = float(self.gaussian_factor_strength_range[0]), float(self.gaussian_factor_strength_range[1])
        s0 = float(max(0.0, s0))
        s1 = float(max(s0, s1))
        self.gaussian_factor_strength_range = (s0, s1)

        e0, e1 = float(self.gaussian_idiosyncratic_noise_range[0]), float(self.gaussian_idiosyncratic_noise_range[1])
        e0 = float(max(0.0, e0))
        e1 = float(max(e0, e1))
        self.gaussian_idiosyncratic_noise_range = (e0, e1)

        # Feature permutation / mask storage policy
        self.store_feature_permutation = str(getattr(self, "store_feature_permutation", "auto")).lower()
        self.store_relevant_mask = str(getattr(self, "store_relevant_mask", "auto")).lower()
        self.store_feature_permutation_max_p = int(max(1, int(getattr(self, "store_feature_permutation_max_p", 2000))))
        self.store_relevant_mask_max_p = int(max(1, int(getattr(self, "store_relevant_mask_max_p", 2000))))

        # Scaling method names
        self.y_scaling = str(getattr(self, "y_scaling", "none")).lower()
        if self.max_relevant_features < 1:
            raise ValueError("max_relevant_features must be >= 1")

        if self.max_signal_pool_features < 1:
            raise ValueError("max_signal_pool_features must be >= 1")
        # Ensure signal pool is not smaller than the relevant cap.
        self.max_signal_pool_features = int(max(self.max_signal_pool_features, self.max_relevant_features))

        # Clamp beta sparsity range
        b_lo, b_hi = self.beta_sparsity_range
        b_lo = int(max(1, b_lo))
        b_hi = int(max(b_lo, min(b_hi, self.max_relevant_features)))
        hookup = (b_lo, b_hi)
        self.beta_sparsity_range = hookup

        # Clamp tree feature count range
        t_lo, t_hi = self.tree_feature_count_range
        t_lo = int(max(1, t_lo))
        t_hi = int(max(t_lo, min(t_hi, self.max_relevant_features)))
        self.tree_feature_count_range = (t_lo, t_hi)

        # Clamp BNN informative feature range
        m_lo, m_hi = self.bnn_informative_features_range
        m_lo = int(max(1, m_lo))
        m_hi = int(max(m_lo, min(m_hi, self.max_relevant_features)))
        self.bnn_informative_features_range = (m_lo, m_hi)

        # Sanitize generator names
        self.x_generator = str(self.x_generator).lower()
        self.bnn_base_generator = str(self.bnn_base_generator).lower()
        self.y_generator = str(self.y_generator).lower()
        self.task_type = str(self.task_type).lower()
        self.x_scaling = str(self.x_scaling).lower()


class LargePSmallNSynthGenerator:
    """Main generator class that produces one dataset at a time."""

    def __init__(self, cfg: LargePSmallNSynthConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -------------------------
    # Label helpers (classification)
    # -------------------------

    def _sample_n_classes(self, y_hat: np.ndarray) -> int:
        if self.cfg.n_classes is not None:
            return int(self.cfg.n_classes)
        max_c = min(10, int(len(y_hat)))
        return int(self.rng.integers(2, max_c + 1))

    def _continuous_to_multiclass(self, y_hat: np.ndarray) -> np.ndarray:
        """Convert continuous y_hat into multiclass labels by random boundaries + balancing."""
        rng = self.rng
        n = y_hat.shape[0]
        Nc = self._sample_n_classes(y_hat)
        if Nc <= 1:
            return np.zeros(n, dtype=np.int64)

        def assign(bounds: np.ndarray) -> np.ndarray:
            b = np.sort(bounds)
            return (y_hat[:, None] > b[None, :]).sum(axis=1).astype(np.int64)

        y = None
        for _ in range(20):
            bounds = rng.choice(y_hat, size=(Nc - 1), replace=False)
            y_tmp = assign(bounds)
            counts = np.bincount(y_tmp, minlength=Nc)
            if counts.min() >= self.cfg.min_class_count:
                y = y_tmp
                break
        if y is None:
            qs = np.linspace(0, 1, Nc + 1)[1:-1]
            bounds = np.quantile(y_hat, qs)
            y = assign(bounds)
        perm = rng.permutation(Nc)
        return perm[y].astype(np.int64, copy=False)

    # -------------------------
    # Noise utilities for y
    # -------------------------

    def _sample_unit_noise(
        self,
        rng: np.random.Generator,
        n: int,
        dist: str,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample mean-0, var-1 noise with a chosen distribution."""
        dist = dist.lower()
        meta: Dict[str, Any] = {"eps_dist": dist}
        if dist == "normal":
            e = rng.normal(0.0, 1.0, size=n)
            e = (e - e.mean()) / (e.std() + 1e-12)
        elif dist == "laplace":
            e = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=n)
            e = (e - e.mean()) / (e.std() + 1e-12)
        elif dist in ("t", "student_t", "studentt"):
            lo, hi = self.cfg.eps_df_range
            df = int(rng.integers(int(lo), int(hi) + 1))
            z = rng.standard_t(df=df, size=n)
            z = z / np.sqrt(df / (df - 2.0))
            # Tighten meta-SNR vs empirical-SNR: normalize by realized sample std as well.
            z = (z - z.mean()) / (z.std() + 1e-12)
            e = z
            meta["eps_df"] = df
        elif dist == "lognormal":
            z = rng.lognormal(mean=0.0, sigma=1.0, size=n)
            z = (z - z.mean()) / (z.std() + 1e-8)
            e = z
        else:
            raise ValueError(f"Unknown eps_dist: {dist}")

    def _sample_alpha_target(
        self,
        rng: np.random.Generator,
    ) -> Tuple[float, Dict[str, Any]]:
        """Sample alpha in (0,1) for Sync9 alpha-mix noise control.

        By design, R^2_target ≈ alpha and SNR_target ≈ alpha/(1-alpha) after robust-standardization.
        We intentionally bias alpha toward low values (R^2 < 0.3 common) to avoid overly strong signal tasks.
        """
        cfg = self.cfg
        mode = str(getattr(cfg, "alpha_sampling", "mixture")).lower()

        def _sample_uniform(r: Tuple[float, float]) -> float:
            lo, hi = float(r[0]), float(r[1])
            lo = float(max(lo, 1e-6))
            hi = float(min(max(hi, lo), 1.0 - 1e-6))
            return float(rng.uniform(lo, hi))

        if mode == "uniform":
            a = _sample_uniform(cfg.alpha_range)
            return a, {"alpha_sampling": mode, "alpha_target": a}

        if mode == "beta":
            a0, a1 = cfg.alpha_range
            p, q = cfg.alpha_beta_params
            u = float(rng.beta(float(p), float(q)))
            a = float(a0 + u * (a1 - a0))
            a = float(min(max(a, 1e-6), 1.0 - 1e-6))
            return a, {"alpha_sampling": mode, "alpha_target": a, "alpha_beta_p": float(p), "alpha_beta_q": float(q)}

        if mode == "mixture":
            w_mod = float(getattr(cfg, "alpha_mixture_w_moderate", 0.1))
            w_mod = float(min(max(w_mod, 0.0), 1.0))
            comp = "moderate" if rng.random() < w_mod else "low"
            r = cfg.alpha_moderate_range if comp == "moderate" else cfg.alpha_low_range
            a = _sample_uniform(r)
            return a, {"alpha_sampling": mode, "alpha_target": a, "alpha_component": comp, "alpha_weight_moderate": w_mod}

        raise ValueError(f"Unknown alpha_sampling mode: {mode}")

    def _sample_snr_target(
        self,
        rng: np.random.Generator,
        step: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Sample an SNR value according to cfg.snr_sampling.

        Supported modes (all i.i.d. in Sync9):
        - uniform:      snr ~ Uniform(cfg.snr_range)
        - log_uniform:  log(snr) ~ Uniform(log(lo), log(hi))
        - mixture:      mixture of moderate vs low SNR components with a *fixed* mixture weight
        - mixture_curriculum: accepted for backward-compat, but treated the same as "mixture"
          (step is ignored to preserve i.i.d. sampling).
        """
        cfg = self.cfg
        mode = str(getattr(cfg, "snr_sampling", "uniform")).lower()

        def _sample_from_range(lo: float, hi: float, *, log_uniform: bool) -> Tuple[float, str]:
            lo = float(max(lo, 1e-8))
            hi = float(max(hi, lo))
            if log_uniform:
                snr = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                return snr, "log_uniform"
            snr = float(rng.uniform(lo, hi))
            return snr, "uniform"

        if mode == "uniform":
            snr, dist = _sample_from_range(cfg.snr_range[0], cfg.snr_range[1], log_uniform=False)
            return snr, {"snr_sampling": mode, "snr_dist": dist}

        if mode == "log_uniform":
            snr, dist = _sample_from_range(cfg.snr_range[0], cfg.snr_range[1], log_uniform=True)
            return snr, {"snr_sampling": mode, "snr_dist": dist}

        if mode in ("mixture", "mixture_curriculum"):
            # NOTE: In Sync9 we remove step-dependent drift; this is an i.i.d. mixture.
            w_mod = float(getattr(cfg, "snr_mixture_w_moderate_start", 0.5))
            w_mod = float(min(max(w_mod, 0.0), 1.0))

            comp = "moderate" if rng.random() < w_mod else "low"
            if comp == "moderate":
                lo, hi = cfg.snr_moderate_range
            else:
                lo, hi = cfg.snr_low_range

            logu = bool(getattr(cfg, "snr_mixture_log_uniform", True))
            snr, dist = _sample_from_range(lo, hi, log_uniform=logu)

            meta = {
                "snr_sampling": ("mixture" if mode == "mixture_curriculum" else mode),
                "snr_component": comp,
                "snr_weight_moderate": w_mod,
                "snr_dist": dist,
            }
            if mode == "mixture_curriculum":
                meta["snr_curriculum_removed"] = True
                meta["snr_curriculum_step_ignored"] = (None if step is None else int(step))
            return snr, meta

        raise ValueError(f"Unknown snr_sampling mode: {mode}")


    def _add_eps_by_snr(
        self,
        rng: np.random.Generator,
        y_clean: np.ndarray,
        *,
        step: Optional[int] = None,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return (y, y_hat_used, meta) by adding noise.

        Sync9 default behavior (cfg.noise_control == "alpha_mix"):
          1) robust-standardize f = y_clean and raw eps using median/MAD
          2) mix by alpha:
                y_hat_used = sqrt(alpha) * f_tilde
                y          = y_hat_used + sqrt(1-alpha) * eps_tilde

        This keeps the *scale* of y aligned across different y-generators (SCM/BNN/linear/tree),
        and removes step-dependent curriculum drift.

        Legacy behavior (cfg.noise_control == "snr_scale_eps"):
          y_hat_used = y_clean
          y          = y_clean + sigma * eps_unit, where sigma is chosen to match sampled SNR.
        """
        cfg = self.cfg
        control = str(getattr(cfg, "noise_control", "alpha_mix")).lower()

        n = int(y_clean.shape[0])
        y_clean64 = y_clean.astype(np.float64, copy=False)

        # ------------------------------------------------------------------
        # Helper: sample raw eps from a distribution (no centering/scaling)
        # ------------------------------------------------------------------
        def _sample_eps_raw(dist: str) -> Tuple[np.ndarray, Dict[str, Any]]:
            dist = dist.lower()
            meta: Dict[str, Any] = {"eps_dist": dist}
            if dist == "normal":
                e = rng.normal(0.0, 1.0, size=n)
            elif dist == "laplace":
                # scale chosen so that var=1 for the underlying Laplace
                e = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=n)
            elif dist in ("t", "student_t", "studentt"):
                lo, hi = cfg.eps_df_range
                df = int(rng.integers(int(lo), int(hi) + 1))
                e = rng.standard_t(df=df, size=n)
                meta["eps_df"] = df
            elif dist == "lognormal":
                e = rng.lognormal(mean=0.0, sigma=1.0, size=n)
            else:
                raise ValueError(f"Unknown eps_dist: {dist}")
            return e.astype(np.float64, copy=False), meta

        # ------------------------------------------------------------------
        # Helper: robust standardization (median/MAD)
        # ------------------------------------------------------------------
        def _robust_standardize(v64: np.ndarray, *, prefix: str) -> Tuple[np.ndarray, Dict[str, Any]]:
            eps = float(getattr(cfg, "scale_eps", 1e-6))
            scale = float(getattr(cfg, "y_robust_mad_scale", 1.4826))
            med = float(np.median(v64))
            mad = float(np.median(np.abs(v64 - med)))
            sigma0 = float(scale * mad)
            sigma = float(sigma0 + eps)
            z = (v64 - med) / sigma
            meta = {
                f"{prefix}_median": med,
                f"{prefix}_mad": mad,
                f"{prefix}_sigma0": sigma0,
                f"{prefix}_sigma": sigma,
                f"{prefix}_var": float(np.var(z)),
            }
            return z, meta

        # ------------------------------------------------------------------
        # Sync9: alpha-mix noise control
        # ------------------------------------------------------------------
        if control == "alpha_mix":
            alpha, alpha_meta = self._sample_alpha_target(rng)

            # robust-standardize clean signal and noise separately (per dataset)
            f_tilde, f_meta = _robust_standardize(y_clean64, prefix="f")
            eps_dist = str(rng.choice(cfg.eps_dist_choices))
            eps_raw64, eps_raw_meta = _sample_eps_raw(eps_dist)
            eps_tilde, eps_meta = _robust_standardize(eps_raw64, prefix="eps")

            # mix
            sqrt_a = float(np.sqrt(alpha))
            sqrt_1a = float(np.sqrt(1.0 - alpha))
            f_var = float(f_meta.get("f_var", 1.0))
            eps_var = float(eps_meta.get("eps_var", 1.0))
            # Variance-normalize the two parts so that (approx) R^2_target == alpha even under robust scaling.
            w_f = float(sqrt_a / np.sqrt(max(f_var, 1e-12)))
            w_eps = float(sqrt_1a / np.sqrt(max(eps_var, 1e-12)))

            y_hat_used64 = w_f * f_tilde
            noise_part64 = w_eps * eps_tilde
            y64 = y_hat_used64 + noise_part64

            # Diagnostics
            var_sig = float(np.var(y_hat_used64))
            var_noise = float(np.var(noise_part64))
            var_y = float(np.var(y64))
            snr_emp = float(var_sig / max(var_noise, 1e-12))
            r2_emp = float(var_sig / max(var_y, 1e-12))

            snr_target = float(alpha / max(1.0 - alpha, 1e-12))
            snr_abs_err = float(abs(snr_emp - snr_target)) if np.isfinite(snr_emp) else float("nan")
            snr_rel_err = float(snr_abs_err / max(abs(snr_target), 1e-12)) if np.isfinite(snr_abs_err) else float("nan")

            corr_clean_noisy = float("nan")
            try:
                if (var_sig > 0) and np.isfinite(var_sig) and np.isfinite(var_y) and (var_y > 0):
                    corr_clean_noisy = float(np.corrcoef(y_hat_used64, y64)[0, 1])
            except Exception:
                corr_clean_noisy = float("nan")

            meta: Dict[str, Any] = {
                "noise_control": "alpha_mix",
                "alpha_weight_f": float(w_f),
                "alpha_weight_eps": float(w_eps),
                **alpha_meta,
                **f_meta,
                **eps_meta,
                **eps_raw_meta,
                "snr_target": snr_target,
                "snr": snr_target,
                "snr_empirical": snr_emp,
                "snr_abs_err": snr_abs_err,
                "snr_rel_err": snr_rel_err,
                "r2_target": float(alpha),
                "r2_empirical": r2_emp,
                "var_y_clean_raw": float(np.var(y_clean64)),
                "var_y_hat_used": var_sig,
                "var_noise_part": var_noise,
                "var_y": var_y,
                "corr_clean_noisy": corr_clean_noisy,
                "curriculum_step": (None if step is None else int(step)),
            }
            return y64.astype(dtype, copy=False), y_hat_used64.astype(dtype, copy=False), meta

        # ------------------------------------------------------------------
        # Legacy: sigma scaling by SNR (kept for backward compat)
        # ------------------------------------------------------------------
        if control in ("snr_scale_eps", "legacy", "snr"):
            snr_target, snr_meta = self._sample_snr_target(rng, step=step)

            # Use eps distribution choice, and standardize by sample mean/std for a tight SNR match.
            eps_dist = str(rng.choice(cfg.eps_dist_choices))
            eps_unit, eps_meta = self._sample_unit_noise(rng, n, eps_dist, dtype=dtype)

            var_sig_raw = float(np.var(y_clean64))
            var_sig_used = var_sig_raw
            var_sig_was_floored = False
            if (not np.isfinite(var_sig_used)) or (var_sig_used < 1e-12):
                var_sig_used = 1.0
                var_sig_was_floored = True

            snr_safe = float(max(snr_target, 1e-12))
            sigma_raw = float(np.sqrt(var_sig_used / snr_safe))

            eps = (sigma_raw * eps_unit.astype(np.float64, copy=False)).astype(np.float64, copy=False)
            y64 = y_clean64 + eps

            var_eps = float(np.var(eps))
            snr_emp = float(var_sig_raw / max(var_eps, 1e-12)) if np.isfinite(var_sig_raw) else float("nan")
            snr_abs_err = float(abs(snr_emp - snr_target)) if np.isfinite(snr_emp) else float("nan")
            snr_rel_err = float(snr_abs_err / max(abs(snr_target), 1e-12)) if np.isfinite(snr_abs_err) else float("nan")

            corr_clean_noisy = float("nan")
            try:
                if np.isfinite(var_sig_raw) and var_sig_raw > 0 and np.isfinite(var_eps) and var_eps > 0:
                    corr_clean_noisy = float(np.corrcoef(y_clean64, y64)[0, 1])
            except Exception:
                corr_clean_noisy = float("nan")

            meta = {
                "noise_control": "snr_scale_eps",
                **snr_meta,
                "snr_target": float(snr_target),
                "snr": float(snr_target),
                "snr_empirical": snr_emp,
                "snr_abs_err": snr_abs_err,
                "snr_rel_err": snr_rel_err,
                "var_y_clean_raw": float(var_sig_raw),
                "var_eps": float(var_eps),
                "eps_dist": eps_meta.get("eps_dist", eps_dist),
                "eps_sigma_raw": float(sigma_raw),
                "eps_sigma": float(sigma_raw),
                "eps_unit_var": float(np.var(eps_unit.astype(np.float64, copy=False))),
                "var_y_clean_was_floored": bool(var_sig_was_floored),
                "corr_clean_noisy": corr_clean_noisy,
                "curriculum_step": (None if step is None else int(step)),
            }
            if "eps_df" in eps_meta:
                meta["eps_df"] = eps_meta["eps_df"]
            return y64.astype(dtype, copy=False), y_clean.astype(dtype, copy=False), meta

        raise ValueError(f"Unknown noise_control: {control}")

    # -------------------------
    # y generators (sparse linear / tree)
    # -------------------------

    def _sample_sparse_beta(
        self,
        rng: np.random.Generator,
        P: int,
        signal_idx: np.ndarray,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Sample sparse beta (support + values) for y = X beta + eps."""
        cfg = self.cfg

        s_lo, s_hi = cfg.beta_sparsity_range
        s = int(rng.integers(int(s_lo), int(s_hi) + 1))
        s = max(1, min(s, cfg.max_relevant_features))

        support_pool = cfg.beta_support_pool.lower()
        if support_pool == "signal":
            pool = np.asarray(signal_idx, dtype=np.int64)
        elif support_pool == "all":
            pool = np.arange(P, dtype=np.int64)
        elif support_pool == "mixed":
            frac = float(cfg.beta_signal_fraction)
            s_sig = int(round(s * frac))
            s_sig = max(0, min(s_sig, int(len(signal_idx))))
            s_noise = s - s_sig

            sig = np.asarray(signal_idx, dtype=np.int64)
            sup_sig = rng.choice(sig, size=s_sig, replace=False) if s_sig > 0 else np.empty(0, dtype=np.int64)

            mask = np.ones(P, dtype=bool)
            mask[sig] = False
            noise_pool = np.where(mask)[0].astype(np.int64, copy=False)
            sup_noise = rng.choice(noise_pool, size=s_noise, replace=False) if s_noise > 0 else np.empty(0, dtype=np.int64)

            support = np.concatenate([sup_sig, sup_noise]).astype(np.int64, copy=False)
            rng.shuffle(support)
            pool = support
        else:
            raise ValueError(f"Unknown beta_support_pool: {cfg.beta_support_pool}")

        if support_pool in ("signal", "all"):
            if len(pool) < s:
                s = int(len(pool))
            support = rng.choice(pool, size=s, replace=False)
        else:
            support = pool

        support = np.asarray(support, dtype=np.int64)
        support = np.unique(support)
        if support.size > cfg.max_relevant_features:
            support = support[: cfg.max_relevant_features]

        # Coefficients
        coef_scale = float(rng.uniform(float(cfg.beta_coef_scale_range[0]), float(cfg.beta_coef_scale_range[1])))
        coef_dist = cfg.beta_coef_dist.lower()
        if coef_dist == "normal":
            vals = rng.normal(0.0, coef_scale, size=support.size)
        elif coef_dist == "laplace":
            vals = rng.laplace(0.0, coef_scale / np.sqrt(2.0), size=support.size)
        elif coef_dist in ("t", "student_t", "studentt"):
            lo, hi = self.cfg.eps_df_range
            df = int(rng.integers(int(lo), int(hi) + 1))
            z = rng.standard_t(df=df, size=support.size)
            z = z / np.sqrt(df / (df - 2.0))
            vals = coef_scale * z
        else:
            raise ValueError(f"Unknown beta_coef_dist: {cfg.beta_coef_dist}")

        vals = np.asarray(vals, dtype=dtype)
        meta: Dict[str, Any] = {
            "beta_support": support.astype(np.int64, copy=False),
            "beta_values": vals.astype(dtype, copy=False),
            "beta_support_pool": support_pool,
            "beta_coef_dist": coef_dist,
            "beta_coef_scale": coef_scale,
        }
        return support, vals, meta

    def _make_y_linear_sparse(
        self,
        rng: np.random.Generator,
        X: np.ndarray,
        signal_idx: np.ndarray,
        *,
        step: Optional[int] = None,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate regression targets via sparse linear teacher.

        We always generate a *sparse* linear signal first:
            y_clean_raw = X beta   (no intercept)

        Then we add noise via `_add_eps_by_snr`, which may optionally rescale/center the signal
        (Sync9 alpha-mix control) before mixing with noise. We return:

            y         : noisy target
            y_hat     : the "clean signal used for mixing" (y_hat_used)
            meta      : beta info + noise diagnostics + (optional) beta scaling for y_hat_used
        """
        P = int(X.shape[1])

        support, vals, beta_meta = self._sample_sparse_beta(rng, P, signal_idx, dtype=dtype)
        y_clean_raw = X[:, support].dot(vals).astype(dtype, copy=False)

        y_noisy, y_hat_used, eps_meta = self._add_eps_by_snr(rng, y_clean_raw, step=step, dtype=dtype)

        meta: Dict[str, Any] = {**beta_meta, **eps_meta}
        meta["p0"] = int(support.size)

        # If Sync9 alpha-mix was used, y_hat_used is an affine rescaling of y_clean_raw:
        #   y_hat_used = sqrt(alpha) * (y_clean_raw - med) / sigma
        # => beta values also rescale by sqrt(alpha)/sigma, plus an intercept term.
        if str(meta.get("noise_control", "")).lower() == "alpha_mix":
            try:
                alpha = float(meta["alpha_target"])
                f_sigma = float(meta["f_sigma"])
                f_median = float(meta["f_median"])

                f_var = float(meta.get("f_var", 1.0))
                scale_mult = float(np.sqrt(alpha) / (np.sqrt(max(f_var, 1e-12)) * max(f_sigma, 1e-12)))
                meta["beta_values_scaled_for_y_hat_used"] = (scale_mult * vals.astype(np.float64)).astype(dtype, copy=False)
                meta["beta_intercept_for_y_hat_used"] = float(-np.sqrt(alpha) * f_median / (np.sqrt(max(f_var, 1e-12)) * max(f_sigma, 1e-12)))
                meta["beta_scale_multiplier_for_y_hat_used"] = float(scale_mult)
                meta["y_hat_is_scaled_signal"] = True
            except Exception:
                meta["y_hat_is_scaled_signal"] = True
                meta["beta_scale_failed"] = True
        else:
            meta["y_hat_is_scaled_signal"] = False

        return y_noisy.astype(dtype, copy=False), y_hat_used.astype(dtype, copy=False), meta


    def _random_tree_predict(
        self,
        rng: np.random.Generator,
        X_sub: np.ndarray,
        max_depth: int,
        min_leaf: int,
        leaf_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random axis-aligned decision tree; return predictions and used local feature ids."""
        n, m = X_sub.shape
        pred = np.empty(n, dtype=X_sub.dtype)
        used_local: List[int] = []
        stack: List[Tuple[np.ndarray, int]] = [(np.arange(n, dtype=np.int64), 0)]
        while stack:
            idx, d = stack.pop()
            if d >= max_depth or idx.size <= min_leaf:
                pred[idx] = rng.normal(0.0, leaf_scale)
                continue
            f = int(rng.integers(0, m))
            col = X_sub[idx, f]
            uniq = np.unique(col)
            if uniq.size <= 1:
                pred[idx] = rng.normal(0.0, leaf_scale)
                continue
            thr = float(rng.choice(uniq[:-1]))
            left = idx[col <= thr]
            right = idx[col > thr]
            if left.size == 0 or right.size == 0:
                pred[idx] = rng.normal(0.0, leaf_scale)
                continue
            used_local.append(f)
            stack.append((left, d + 1))
            stack.append((right, d + 1))
        used = np.unique(np.asarray(used_local, dtype=np.int64))
        return pred.astype(X_sub.dtype, copy=False), used

    def _make_y_tree_like(
        self,
        rng: np.random.Generator,
        X: np.ndarray,
        signal_idx: np.ndarray,
        kind: str,
        *,
        step: Optional[int] = None,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate regression targets via a sampled RF/GBDT-like teacher: y = f_tree(X) + eps.

        Returns:
          y: noisy target
          y_hat: clean teacher signal f_tree(X)
          meta: teacher + SNR/noise diagnostics
        """
        cfg = self.cfg
        kind = kind.lower()
        if kind not in ("rf_like", "gbdt_like"):
            raise ValueError(f"Unknown tree kind: {kind}")

        N, P = X.shape

        # Choose candidate features for the teacher (hard-clamped to <= max_relevant_features)
        m_lo, m_hi = cfg.tree_feature_count_range
        m = int(rng.integers(int(m_lo), int(m_hi) + 1))
        m = max(1, min(int(P), min(m, cfg.max_relevant_features)))

        sig = np.asarray(signal_idx, dtype=np.int64)
        m_sig = int(round(m * float(cfg.tree_signal_fraction)))
        m_sig = max(0, min(m_sig, int(sig.size)))
        m_noise = m - m_sig

        feat_parts: List[np.ndarray] = []
        if m_sig > 0:
            feat_parts.append(rng.choice(sig, size=m_sig, replace=False).astype(np.int64, copy=False))
        if m_noise > 0:
            mask = np.ones(int(P), dtype=bool)
            mask[sig] = False
            noise_pool = np.where(mask)[0].astype(np.int64, copy=False)
            feat_parts.append(rng.choice(noise_pool, size=m_noise, replace=False).astype(np.int64, copy=False))
        feat_idx = np.concatenate(feat_parts) if feat_parts else rng.choice(np.arange(P), size=1, replace=False)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        rng.shuffle(feat_idx)
        feat_idx = np.unique(feat_idx)[:m]

        X_sub = X[:, feat_idx]

        # Ensemble hyper-parameters
        t_lo, t_hi = cfg.tree_n_estimators_range
        n_estimators = int(rng.integers(int(t_lo), int(t_hi) + 1))
        n_estimators = max(1, n_estimators)

        d_lo, d_hi = cfg.tree_max_depth_range
        max_depth = int(rng.integers(int(d_lo), int(d_hi) + 1))
        max_depth = max(1, max_depth)

        min_leaf = int(max(1, cfg.tree_min_leaf))
        leaf_scale = float(cfg.tree_leaf_scale)

        lr = None
        if kind == "gbdt_like":
            lr = float(rng.uniform(float(cfg.gbdt_learning_rate_range[0]), float(cfg.gbdt_learning_rate_range[1])))

        # Build the teacher function f(X)
        used_union: List[int] = []
        f = np.zeros(int(N), dtype=dtype)
        for _ in range(int(n_estimators)):
            pred_t, used_local = self._random_tree_predict(rng, X_sub, max_depth=max_depth, min_leaf=min_leaf, leaf_scale=leaf_scale)
            used_union.append(used_local)
            if kind == "rf_like":
                f += pred_t.astype(dtype, copy=False)
            else:
                assert lr is not None
                f += float(lr) * pred_t.astype(dtype, copy=False)
        if kind == "rf_like":
            f = f / float(n_estimators)

        if used_union:
            used_union = np.unique(np.concatenate(used_union).astype(np.int64, copy=False))
        else:
            used_union = np.empty(0, dtype=np.int64)

        if used_union.size:
            tree_used = np.sort(feat_idx[used_union])
        else:
            tree_used = np.sort(feat_idx[:1])

        tree_used = tree_used[: cfg.max_relevant_features]

        y_clean = f.astype(dtype, copy=False)
        y_noisy, y_hat_used, eps_meta = self._add_eps_by_snr(rng, y_clean, step=step, dtype=dtype)

        meta: Dict[str, Any] = {
            "tree_type": kind,
            "tree_feature_candidates": np.sort(feat_idx).astype(np.int64, copy=False),
            "tree_feature_used": tree_used.astype(np.int64, copy=False),
            "tree_n_estimators": n_estimators,
            "tree_max_depth": max_depth,
            "tree_min_leaf": min_leaf,
            "tree_leaf_scale": leaf_scale,
            **eps_meta,
        }
        if lr is not None:
            meta["gbdt_learning_rate"] = float(lr)

        return y_noisy.astype(dtype, copy=False), y_hat_used.astype(dtype, copy=False), meta

    # -------------------------
    # X generators (scm / gaussian / bnn-overwrite)
    # -------------------------

    
    def _standardize_X(
        self,
        rng: np.random.Generator,
        X: np.ndarray,
        *,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Column-wise scaling for X using only training/context samples.

        Controlled by cfg.x_scaling:
          - "standard": z-score using mean/std computed on the first cfg.n_train rows
          - "robust": robust z-score using median/MAD (scaled by cfg.robust_mad_scale), stats on first cfg.n_train rows
          - "none": return X unchanged

        Why "train-only"?
          In PFN-style training, preprocessing statistics must be computed only from the training/context
          samples (otherwise we leak information from the query/test samples). This matches TabPFN's
          preprocessing protocol.

        Returns:
          X_scaled, scale_meta
        """
        cfg = self.cfg
        method = str(getattr(cfg, "x_scaling", "standard")).lower()

        N = int(X.shape[0])
        n_train = int(getattr(cfg, "n_train", N))
        n_train = int(max(1, min(n_train, N)))
        eps = float(getattr(cfg, "scale_eps", 1e-6))

        scale_meta: Dict[str, Any] = {
            "x_scaling": method,
            "x_scale_n_train": n_train,
        }

        if method in ("none", "identity", "no", "raw"):
            return X.astype(dtype, copy=False), scale_meta

        X_train = X[:n_train]
        small_idx = np.empty(0, dtype=np.int64)

        if method == "robust":
            med = np.median(X_train, axis=0, keepdims=True)
            mad = np.median(np.abs(X_train - med), axis=0, keepdims=True)
            scale = float(getattr(cfg, "robust_mad_scale", 1.4826))
            sigma0 = scale * mad
            small = sigma0 < eps
            if np.any(small):
                idx = np.where(small.ravel())[0]
                small_idx = idx
                scale_meta["x_scale_small_sigma_count"] = int(idx.size)
                scale_meta["x_scale_small_sigma_examples"] = idx[:10].astype(np.int64, copy=False)
            sigma = sigma0 + eps
            Xs = (X - med) / sigma
        else:
            mu = X_train.mean(axis=0, keepdims=True)
            sigma0 = X_train.std(axis=0, keepdims=True)
            small = sigma0 < eps
            if np.any(small):
                idx = np.where(small.ravel())[0]
                small_idx = idx
                scale_meta["x_scale_small_sigma_count"] = int(idx.size)
                scale_meta["x_scale_small_sigma_examples"] = idx[:10].astype(np.int64, copy=False)
            sigma = sigma0 + eps
            Xs = (X - mu) / sigma

        # Optional: tiny jitter to near-constant columns to avoid NaNs / line artifacts in corr heatmaps.
        if bool(getattr(cfg, "fix_near_constant_X", True)) and (small_idx.size > 0):
            idx = small_idx.astype(np.int64, copy=False)
            jitter_std = float(getattr(cfg, "near_constant_jitter_std", 1e-3))
            Xs[:, idx] = Xs[:, idx] + rng.normal(0.0, jitter_std, size=(N, idx.size)).astype(dtype, copy=False)
            scale_meta["x_scale_jitter_std"] = float(jitter_std)

        return Xs.astype(dtype, copy=False), scale_meta

    def _scale_y_pair(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        *,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Scale regression targets (y and y_hat) using train-only statistics.

        Stats are computed on the first cfg.n_train samples of *y* (observed target),
        then applied to both y and y_hat for scale consistency.

        This is helpful for:
          - stable NN training when y is an input (feature selection nets, PFN-style models)
          - avoiding outlier-driven scale blow-ups (robust MAD option)

        Returns:
          y_scaled, y_hat_scaled, y_scale_meta
        """
        cfg = self.cfg
        method = str(getattr(cfg, "y_scaling", "none")).lower()

        N = int(y.shape[0])
        n_train = int(getattr(cfg, "n_train", N))
        n_train = int(max(1, min(n_train, N)))
        eps = float(getattr(cfg, "scale_eps", 1e-6))

        meta: Dict[str, Any] = {"y_scaling": method, "y_scale_n_train": n_train}

        if method in ("none", "identity", "no", "raw"):
            return y.astype(dtype, copy=False), y_hat.astype(dtype, copy=False), meta

        y_train = y[:n_train].astype(np.float64, copy=False)

        if method == "robust":
            med = float(np.median(y_train))
            mad = float(np.median(np.abs(y_train - med)))
            scale = float(getattr(cfg, "y_robust_mad_scale", 1.4826))
            sigma0 = float(scale * mad)
            sigma = float(sigma0 + eps)
            meta.update(
                {
                    "y_scale_median": med,
                    "y_scale_mad": mad,
                    "y_scale_sigma0": sigma0,
                    "y_scale_sigma": sigma,
                }
            )
            y_s = (y.astype(np.float64, copy=False) - med) / sigma
            yhat_s = (y_hat.astype(np.float64, copy=False) - med) / sigma
        else:
            mu = float(np.mean(y_train))
            sigma0 = float(np.std(y_train))
            sigma = float(sigma0 + eps)
            meta.update(
                {
                    "y_scale_mean": mu,
                    "y_scale_std": sigma0,
                    "y_scale_sigma0": sigma0,
                    "y_scale_sigma": sigma,
                }
            )
            y_s = (y.astype(np.float64, copy=False) - mu) / sigma
            yhat_s = (y_hat.astype(np.float64, copy=False) - mu) / sigma

        if sigma0 < eps:
            meta["y_scale_small_sigma"] = True

        return y_s.astype(dtype, copy=False), yhat_s.astype(dtype, copy=False), meta

    def _generate_X_gaussian(
        self,
        rng: np.random.Generator,
        N: int,
        P: int,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate X from a Gaussian prior.

        Sync9 default (cfg.gaussian_corr == "block_factor"):
          blockwise correlated Gaussian features via a low-rank factor model inside each block.

        This addresses the "iid Gaussian is too easy / unrealistic" concern and produces
        correlated feature groups (harder screening / more realistic collinearity).
        """
        cfg = self.cfg
        corr = str(getattr(cfg, "gaussian_corr", "block_factor")).lower()

        meta: Dict[str, Any] = {"x_mechanism": "gaussian", "gaussian_corr": corr}

        if corr in ("independent", "iid", "none"):
            X = rng.normal(0.0, 1.0, size=(N, P)).astype(dtype, copy=False)
            X, scale_meta = self._standardize_X(rng, X, dtype=dtype)
            meta.update(scale_meta)
            meta["x_mechanism"] = "gaussian_iid"
            return X, meta

        if corr in ("block_factor", "block", "blockwise", "factor"):
            bmin, bmax = cfg.gaussian_block_size_range
            rmin, rmax = cfg.gaussian_factor_rank_range

            sizes: List[int] = []
            rem = int(P)
            while rem > 0:
                b = int(rng.integers(int(bmin), int(bmax) + 1))
                b = int(min(b, rem))
                b = int(max(2, b))
                sizes.append(b)
                rem -= b

            blocks: List[np.ndarray] = []
            ranks: List[int] = []
            strengths: List[float] = []
            noises: List[float] = []

            for b in sizes:
                r_hi = int(min(int(rmax), int(b)))
                r_lo = int(min(int(rmin), r_hi))
                r = int(rng.integers(r_lo, r_hi + 1))
                s = float(rng.uniform(float(cfg.gaussian_factor_strength_range[0]), float(cfg.gaussian_factor_strength_range[1])))
                sig = float(rng.uniform(float(cfg.gaussian_idiosyncratic_noise_range[0]), float(cfg.gaussian_idiosyncratic_noise_range[1])))

                U = rng.normal(0.0, 1.0, size=(int(N), int(r)))
                L = rng.normal(0.0, 1.0, size=(int(b), int(r)))
                E = rng.normal(0.0, 1.0, size=(int(N), int(b)))

                Xb = (s * (U @ L.T) + sig * E).astype(dtype, copy=False)

                if bool(getattr(cfg, "gaussian_shuffle_within_block", True)) and b > 1:
                    perm_b = rng.permutation(int(b))
                    Xb = Xb[:, perm_b]

                blocks.append(Xb)
                ranks.append(int(r))
                strengths.append(float(s))
                noises.append(float(sig))

            block_order = list(range(len(blocks)))
            if bool(getattr(cfg, "gaussian_shuffle_blocks", True)) and len(blocks) > 1:
                order = rng.permutation(len(blocks)).astype(int)
                blocks = [blocks[i] for i in order]
                sizes = [sizes[i] for i in order]
                ranks = [ranks[i] for i in order]
                strengths = [strengths[i] for i in order]
                noises = [noises[i] for i in order]
                block_order = order.tolist()

            X = np.concatenate(blocks, axis=1).astype(dtype, copy=False)
            X, scale_meta = self._standardize_X(rng, X, dtype=dtype)

            # Summary meta (keep it small)
            meta.update(scale_meta)
            meta.update(
                {
                    "x_mechanism": "gaussian_block_factor",
                    "gaussian_block_count": int(len(sizes)),
                    "gaussian_block_size_min": int(min(sizes)) if sizes else int(P),
                    "gaussian_block_size_max": int(max(sizes)) if sizes else int(P),
                    "gaussian_block_size_mean": float(np.mean(sizes)) if sizes else float(P),
                    "gaussian_rank_min": int(min(ranks)) if ranks else 0,
                    "gaussian_rank_max": int(max(ranks)) if ranks else 0,
                    "gaussian_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
                    "gaussian_factor_strength_mean": float(np.mean(strengths)) if strengths else 0.0,
                    "gaussian_idiosyncratic_noise_mean": float(np.mean(noises)) if noises else 0.0,
                    "gaussian_block_order_head": block_order[:10],
                    "gaussian_block_sizes_head": sizes[:10],
                    "gaussian_ranks_head": ranks[:10],
                }
            )
            return X, meta

        raise ValueError(f"Unknown gaussian_corr: {corr}")


    def _generate_X_scm(
        self,
        rng: np.random.Generator,
        N: int,
        P: int,
        act,
        act_name: str,
        signal_mask: np.ndarray,
        dtype=np.float32,
        return_graph: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Generate X using the sparse SCM-style layered MLP prior."""
        cfg = self.cfg

        # Build sparse DAG for hidden layers
        layer_dims = (cfg.latent_dim,) + tuple(cfg.hidden_dims)
        Ws_hidden: List[sp.csr_matrix] = []
        in_dim = layer_dims[0]
        for out_dim in layer_dims[1:]:
            W = build_sparse_csr(
                rng=rng,
                out_dim=out_dim,
                in_dim=in_dim,
                fan_in=cfg.fan_in_hidden,
                weight_std=cfg.weight_std / np.sqrt(cfg.fan_in_hidden),
                dropout=cfg.dropout_hidden,
                dtype=dtype,
            )
            Ws_hidden.append(W)
            in_dim = out_dim

        last_hidden_dim = layer_dims[-1]
        label_latent_dim = int(min(cfg.label_latent_dim, last_hidden_dim))
        label_latent_dim = max(1, label_latent_dim)

        # Parent index ranges: signal features connect to label_latent_dim subspace
        if label_latent_dim == last_hidden_dim:
            row_ranges = (signal_mask, (0, last_hidden_dim), (0, last_hidden_dim))
        else:
            row_ranges = (signal_mask, (0, label_latent_dim), (label_latent_dim, last_hidden_dim))

        W_feat = build_sparse_csr(
            rng=rng,
            out_dim=P,
            in_dim=last_hidden_dim,
            fan_in=cfg.fan_in_features,
            weight_std=cfg.weight_std / np.sqrt(cfg.fan_in_features),
            dropout=cfg.dropout_features,
            row_index_ranges=row_ranges,
            dtype=dtype,
        )

        # Forward pass
        Z = _sample_mixed_inputs(rng, N, cfg.latent_dim, dtype=dtype)
        for W in Ws_hidden:
            lin = W.dot(Z.T).T
            if cfg.noise_std_hidden > 0:
                lin = lin + rng.normal(0.0, cfg.noise_std_hidden, size=lin.shape).astype(dtype, copy=False)
            Z = act(lin).astype(dtype, copy=False)
        H = Z

        X_lin = sparse_linear_in_chunks(
            W=W_feat,
            H=H,
            block_rows=cfg.feature_block_rows,
            n_jobs=cfg.n_jobs_features,
            dtype=dtype,
        )
        if cfg.noise_std_features > 0:
            X_lin = X_lin + rng.normal(0.0, cfg.noise_std_features, size=X_lin.shape).astype(dtype, copy=False)
        X = act(X_lin).astype(dtype, copy=False)
        X, scale_meta = self._standardize_X(rng, X, dtype=dtype)

        meta = {**scale_meta, 
            "x_mechanism": "scm",
            "activation": act_name,
            "label_latent_dim": int(label_latent_dim),
        }

        graph = None
        if return_graph:
            graph = {"Ws_hidden": Ws_hidden, "W_feat": W_feat}

        return X, H, meta, graph

    def _generate_X_bnn_overwrite(
        self,
        rng: np.random.Generator,
        N: int,
        P: int,
        act,
        act_name: str,
        signal_idx: np.ndarray,
        signal_mask: np.ndarray,
        dtype=np.float32,
        return_graph: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Generate X by overwriting m columns with BNN-generated informative features."""
        cfg = self.cfg

        # 1) Generate a large base X (gaussian or scm)
        base = cfg.bnn_base_generator
        H_base: Optional[np.ndarray] = None
        graph = None
        base_meta: Dict[str, Any]
        if base == "gaussian":
            X_base, base_meta = self._generate_X_gaussian(rng, N, P, dtype=dtype)
        elif base == "scm":
            X_base, H_base, base_meta, graph = self._generate_X_scm(
                rng,
                N,
                P,
                act=act,
                act_name=act_name,
                signal_mask=signal_mask,
                dtype=dtype,
                return_graph=return_graph,
            )
        else:
            raise ValueError(f"Unknown bnn_base_generator: {cfg.bnn_base_generator}")

        # 2) Decide how many informative features to overwrite (m <= max_relevant_features)
        m_lo, m_hi = cfg.bnn_informative_features_range
        m = int(rng.integers(int(m_lo), int(m_hi) + 1))
        m = max(1, min(m, cfg.max_relevant_features, P))
        if signal_idx.size < m:
            m = int(signal_idx.size)
        if m < 1:
            m = 1

        overwrite_idx = rng.choice(signal_idx, size=m, replace=False).astype(np.int64, copy=False)
        overwrite_idx = np.sort(overwrite_idx)

        # 3) Generate m informative features via a small random BNN
        X_inf, bnn_meta = self._generate_BNN_features(rng, N, m, act, act_name, dtype=dtype)

        # 4) Overwrite and re-standardize
        X_base[:, overwrite_idx] = X_inf
        X, scale_meta = self._standardize_X(rng, X_base, dtype=dtype)

        # IMPORTANT: build meta carefully so x_mechanism reflects the final mechanism
        # (dictionary unpacking order would otherwise overwrite it).
        meta = dict(base_meta)
        meta['bnn_base_x_mechanism'] = base_meta.get('x_mechanism')
        meta.update(bnn_meta)
        meta.update(scale_meta)
        meta.update(
            {
                'x_mechanism': 'bnn_overwrite',
                'bnn_base_generator': base,
                'bnn_overwrite_indices': overwrite_idx.astype(np.int64, copy=False),
                'bnn_m': int(m),
            }
        )
        return X, H_base, meta, graph

    def _generate_BNN_features(
        self,
        rng: np.random.Generator,
        N: int,
        m: int,
        act,
        act_name: str,
        dtype=np.float32,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate m informative features using a random (Bayesian-style) MLP."""
        cfg = self.cfg

        # Input is m-dimensional (as discussed: start from N x m, then map to N x m)
        X0 = _sample_mixed_inputs(rng, N, m, dtype=dtype)

        # Feature-wise scaling amplifies irregularity (lightweight analogue of TabPFN refinements)
        s_lo, s_hi = cfg.bnn_input_scale_range
        scales = rng.uniform(float(s_lo), float(s_hi), size=m).astype(dtype, copy=False)
        X = X0 * scales[None, :]

        # Build and apply random dense layers (m is small, so dense is fine)
        dims = (m,) + tuple(cfg.bnn_hidden_dims) + (m,)
        for li in range(len(dims) - 1):
            din, dout = dims[li], dims[li + 1]
            W = rng.normal(0.0, cfg.weight_std / np.sqrt(max(1, din)), size=(din, dout)).astype(dtype, copy=False)

            # Weight dropout (acts like sparsification without CSR overhead at small m)
            if cfg.bnn_weight_dropout > 0:
                keep = (rng.random(size=W.shape) >= cfg.bnn_weight_dropout).astype(dtype, copy=False)
                W = W * keep

            X = X @ W
            if cfg.bnn_noise_std > 0:
                X = X + rng.normal(0.0, cfg.bnn_noise_std, size=X.shape).astype(dtype, copy=False)

            # Use activation on hidden layers, but keep output layer as identity
            if li < len(dims) - 2:
                X = act(X).astype(dtype, copy=False)

        X, scale_meta = self._standardize_X(rng, X, dtype=dtype)
        meta = {**scale_meta, 
            "bnn_activation": act_name,
            "bnn_hidden_dims": tuple(int(x) for x in cfg.bnn_hidden_dims),
            "bnn_weight_dropout": float(cfg.bnn_weight_dropout),
            "bnn_noise_std": float(cfg.bnn_noise_std),
        }
        return X.astype(dtype, copy=False), meta

    # -------------------------
    # Public API
    # -------------------------

    def generate_one(self, seed: Optional[int] = None, *, step: Optional[int] = None, return_graph: bool = False) -> Dict[str, Any]:
        """Generate one dataset dict: {X, y, y_hat, meta} (and optionally graph)."""
        cfg = self.cfg
        seed_used = int(seed) if seed is not None else (int(self.seed) if self.seed is not None else None)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))

        P, N = int(cfg.n_features), int(cfg.n_samples)
        dtype = cfg.dtype

        # Sample activation per dataset
        act_name = str(rng.choice(cfg.activation_choices))
        act = _activation(act_name)

        # Sample a pool of "signal" features (can be larger than the <=max_relevant "true relevant" cap)
        n_signal = int(round(P * float(cfg.signal_feature_fraction)))
        n_signal = max(1, min(P, min(n_signal, cfg.max_signal_pool_features)))
        signal_idx = rng.choice(P, size=n_signal, replace=False).astype(np.int64, copy=False)
        signal_idx = np.sort(signal_idx)
        signal_mask = np.zeros(P, dtype=bool)
        signal_mask[signal_idx] = True

        # Generate X according to the chosen mechanism
        x_mech = cfg.x_generator
        H: Optional[np.ndarray] = None
        graph = None
        x_meta: Dict[str, Any]

        if x_mech == "scm":
            X, H, x_meta, graph = self._generate_X_scm(
                rng,
                N,
                P,
                act=act,
                act_name=act_name,
                signal_mask=signal_mask,
                dtype=dtype,
                return_graph=return_graph,
            )
        elif x_mech == "gaussian":
            X, x_meta = self._generate_X_gaussian(rng, N, P, dtype=dtype)
        elif x_mech == "bnn":
            X, H, x_meta, graph = self._generate_X_bnn_overwrite(
                rng,
                N,
                P,
                act=act,
                act_name=act_name,
                signal_idx=signal_idx,
                signal_mask=signal_mask,
                dtype=dtype,
                return_graph=return_graph,
            )
        else:
            raise ValueError(f"Unknown x_generator: {cfg.x_generator}")

        # Ensure contiguous memory
        X = np.ascontiguousarray(X)

        # -------------------------
        # Sync9: Feature permutation (remove trivial ordering leakage)
        # -------------------------
        perm_meta: Dict[str, Any] = {}
        if bool(getattr(cfg, "apply_feature_permutation", True)):
            perm = rng.permutation(P).astype(np.int64, copy=False)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(P, dtype=np.int64)

            X = X[:, perm]

            # Update index-based metadata (signal pool)
            signal_idx = inv_perm[signal_idx]
            signal_idx.sort()

            # Update x_meta indices if present (BNN overwrite indices)
            if isinstance(x_meta, dict) and ("bnn_overwrite_indices" in x_meta):
                try:
                    ow = np.asarray(x_meta["bnn_overwrite_indices"], dtype=np.int64)
                    x_meta["bnn_overwrite_indices"] = inv_perm[ow].tolist()
                except Exception:
                    x_meta["bnn_overwrite_indices_perm_failed"] = True

            perm_meta = {
                "feature_permutation_applied": True,
                "feature_permutation_mode": "full_random",
            }

            # Optionally store the permutation (can be large)
            store_perm = str(getattr(cfg, "store_feature_permutation", "auto")).lower()
            max_p = int(getattr(cfg, "store_feature_permutation_max_p", 2000))
            if store_perm == "list" or (store_perm == "auto" and P <= max_p):
                perm_meta["feature_perm"] = perm.tolist()
            elif store_perm == "base64" or (store_perm == "auto" and P <= max_p):
                # base64-encode int32 bytes (compact in JSON)
                try:
                    perm_i32 = perm.astype(np.int32, copy=False)
                    perm_meta["feature_perm_base64_i32"] = base64.b64encode(perm_i32.tobytes()).decode("ascii")
                except Exception:
                    pass
            elif store_perm == "none":
                pass
        else:
            perm_meta = {"feature_permutation_applied": False}

# -------------------------
# Target / label generation
# -------------------------
        task_type = cfg.task_type
        y_mech = cfg.y_generator

        # y_cont: noisy target (continuous), y_hat: clean signal/teacher output (continuous)
        if y_mech == "hidden":
            # Hidden mechanism is meaningful only when X was generated from SCM (we have H).
            if H is None:
                raise ValueError("y_generator='hidden' requires x_generator='scm' (or bnn_base='scm').")

            label_latent_dim = int(x_meta.get("label_latent_dim", H.shape[1]))
            label_latent_dim = max(1, min(label_latent_dim, H.shape[1]))

            parents = rng.integers(0, label_latent_dim, size=max(1, min(cfg.max_relevant_features, 16)))
            w = rng.normal(0.0, cfg.weight_std / np.sqrt(len(parents)), size=len(parents)).astype(dtype)
            y_hat = (H[:, parents] * w[None, :]).sum(axis=1).astype(dtype, copy=False)

            y_cont, y_hat, extra_meta = self._add_eps_by_snr(rng, y_hat, step=step, dtype=dtype)
            relevant_idx = signal_idx
            extra_meta.update({"label_parents": parents.astype(np.int32, copy=False)})

        elif y_mech == "linear_sparse":
            y_cont, y_hat, extra_meta = self._make_y_linear_sparse(rng, X, signal_idx, step=step, dtype=dtype)
            relevant_idx = np.asarray(extra_meta.get("beta_support", signal_idx[:1]), dtype=np.int64)

        elif y_mech in ("rf_like", "gbdt_like"):
            y_cont, y_hat, extra_meta = self._make_y_tree_like(rng, X, signal_idx, kind=y_mech, step=step, dtype=dtype)
            relevant_idx = np.asarray(extra_meta.get("tree_feature_used", signal_idx[:1]), dtype=np.int64)

        else:
            raise ValueError(f"Unknown y_generator: {cfg.y_generator}")

        # Convert to final task target
        if task_type == "classification":
            y = self._continuous_to_multiclass(y_cont.astype(np.float64))
            n_classes = int(y.max() + 1)
        elif task_type == "regression":
            y = y_cont.astype(dtype, copy=False)
            y_hat = y_hat.astype(dtype, copy=False)

            # Train-only scaling of y/y_hat (mean-std or median-MAD)
            y, y_hat, y_scale_meta = self._scale_y_pair(y, y_hat, dtype=dtype)
            extra_meta = {**extra_meta, **y_scale_meta}

            n_classes = None
        else:
            raise ValueError(f"Unknown task_type: {cfg.task_type}")

        # Hard guards (the key "<=max_relevant_features" requirement)
        relevant_idx = np.unique(np.asarray(relevant_idx, dtype=np.int64))
        relevant_idx = np.sort(relevant_idx)[: cfg.max_relevant_features]

        assert signal_idx.size <= cfg.max_signal_pool_features, "signal_feature_indices exceeds max_signal_pool_features"
        assert relevant_idx.size <= cfg.max_relevant_features, "relevant_feature_indices exceeds max_relevant_features"

        # Sync9: store p0 + (optional) dense relevant-feature mask in metadata
        p0 = int(relevant_idx.size)
        mask_meta: Dict[str, Any] = {"p0": p0}
        store_mask = str(getattr(cfg, "store_relevant_mask", "auto")).lower()
        max_p_mask = int(getattr(cfg, "store_relevant_mask_max_p", 2000))
        if store_mask == "dense" or (store_mask == "auto" and P <= max_p_mask):
            try:
                m = np.zeros(P, dtype=np.uint8)
                m[relevant_idx] = 1
                mask_meta["relevant_feature_mask"] = m.tolist()
            except Exception:
                mask_meta["relevant_feature_mask_failed"] = True

        out: Dict[str, Any] = {
            "X": X,
            "y": y,
            "y_hat": y_hat,
            "meta": {
                "x_generator": x_mech,
                "y_mechanism": y_mech,
                "task_type": task_type,
                "n_classes": n_classes,
                "seed": seed_used,
                "curriculum_step": step,
                "n": N,
                "p": P,
                "cfg_snapshot": asdict(cfg),
                "sync_version": SYNC_VERSION,
                **perm_meta,
                **mask_meta,
                "signal_feature_indices": signal_idx.astype(np.int64, copy=False),
                "relevant_feature_indices": relevant_idx.astype(np.int64, copy=False),
                **x_meta,
                **extra_meta,
            },
        }
        if return_graph:
            out["graph"] = graph
        return out


# ---------------------------------
# 6) Memory-safe batch generation
# ---------------------------------

def iter_generate(
    cfg: LargePSmallNSynthConfig,
    num_datasets: int,
    base_seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Yield datasets one-by-one (safe: does not store all datasets in memory)."""
    for i in range(int(num_datasets)):
        seed = int(base_seed + i * 10_000)
        g = LargePSmallNSynthGenerator(cfg, seed=seed)
        yield g.generate_one(seed=seed)


def generate_many(
    cfg: LargePSmallNSynthConfig,
    num_datasets: int,
    n_jobs: int = -1,
    base_seed: int = 0,
    out_dir: Optional[str | Path] = None,
    compress: bool = True,
    overwrite: bool = False,
) -> List[Any]:
    """Generate multiple datasets.

    - If out_dir is None: return a list of dataset dicts (memory heavy).
    - If out_dir is provided: each worker saves immediately, and we return a list of file paths.
    """

    seeds = [int(base_seed + i * 10_000) for i in range(int(num_datasets))]

    if out_dir is None:
        def worker_mem(seed: int) -> Dict[str, Any]:
            g = LargePSmallNSynthGenerator(cfg, seed=seed)
            return g.generate_one(seed=seed)

        return Parallel(n_jobs=n_jobs, prefer="processes")(delayed(worker_mem)(s) for s in seeds)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def worker_save(i: int, seed: int) -> str:
        path = out_dir / f"dataset_{i:06d}_seed{seed}.npz"
        if path.exists() and not overwrite:
            return str(path)
        g = LargePSmallNSynthGenerator(cfg, seed=seed)
        data = g.generate_one(seed=seed)
        save_dataset_npz(path, data["X"], data["y"], data["y_hat"], data["meta"], compress=compress)
        # Free memory aggressively inside the worker
        del data
        gc.collect()
        return str(path)

    return Parallel(n_jobs=n_jobs, prefer="processes")(delayed(worker_save)(i, s) for i, s in enumerate(seeds))


def benchmark_load(paths: Sequence[str | Path], n_jobs: int = 1) -> float:
    """Return wall time (sec) to load all NPZ files."""
    paths = [str(p) for p in paths]

    def _load_one(p: str) -> Tuple[Tuple[int, int], int]:
        d = load_dataset_npz(p)
        X = d["X"]
        y = d["y"]
        shape = (int(X.shape[0]), int(X.shape[1]))
        n_classes = int(np.max(y) + 1) if y.dtype.kind in ("i", "u") else -1
        # Drop references ASAP
        return shape, n_classes

    t0 = time.perf_counter()
    if n_jobs <= 1:
        for p in paths:
            _ = _load_one(p)
    else:
        _ = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_load_one)(p) for p in paths)
    t1 = time.perf_counter()
    return float(t1 - t0)


# ---------------------------------
# 7) Feasibility test runner
# ---------------------------------

class _RSSMonitor:
    """Lightweight RSS monitor using psutil, sampled in a background thread."""

    def __init__(self, interval_sec: float = 0.02):
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.rss0: int = 0
        self.rss_peak: int = 0
        self.rss_end: int = 0

    def __enter__(self) -> "_RSSMonitor":
        p = psutil.Process(os.getpid())
        self.rss0 = int(p.memory_info().rss)
        self.rss_peak = self.rss0

        def _run():
            proc = psutil.Process(os.getpid())
            while not self._stop.is_set():
                rss = int(proc.memory_info().rss)
                if rss > self.rss_peak:
                    self.rss_peak = rss
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        p = psutil.Process(os.getpid())
        self.rss_end = int(p.memory_info().rss)


def run_feasibility(
    cfg_template: LargePSmallNSynthConfig,
    out_dir: str | Path,
    p_list: Sequence[int] = (1_000, 5_000, 10_000, 20_000),
    n_list: Sequence[int] = (50, 100, 500, 1_000),
    repeats: int = 3,
    gen_jobs_list: Sequence[int] = (1, 2, 4, 8),
    batch_size_for_scaling: int = 32,
    compress: bool = True,
) -> Path:
    """Run feasibility tests and write a CSV + Markdown report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "feasibility_results.csv"
    report_md = out_dir / "feasibility_report.md"

    rows: List[Dict[str, Any]] = []

    for P in p_list:
        for N in n_list:
            # Make a per-setting cfg copy
            cfg = LargePSmallNSynthConfig(**{**cfg_template.__dict__})
            cfg.n_features = int(P)
            cfg.n_samples = int(N)
            cfg.__post_init__()

            # Use a deterministic seed per setting for reproducibility
            seed0 = int(1_000_000 + 10_000 * P + N)

            gen_times: List[float] = []
            save_times: List[float] = []
            load_times: List[float] = []
            peak_rss: List[float] = []
            file_sizes: List[float] = []

            for r in range(int(repeats)):
                seed = seed0 + r
                g = LargePSmallNSynthGenerator(cfg, seed=seed)
                tmp_path = out_dir / f"tmp_P{P}_N{N}_r{r}.npz"
                if tmp_path.exists():
                    tmp_path.unlink()

                with _RSSMonitor() as mon:
                    t0 = time.perf_counter()
                    data = g.generate_one(seed=seed)
                    t1 = time.perf_counter()
                    save_dataset_npz(tmp_path, data["X"], data["y"], data["y_hat"], data["meta"], compress=compress)
                    t2 = time.perf_counter()
                    _ = load_dataset_npz(tmp_path)
                    t3 = time.perf_counter()

                gen_times.append(float(t1 - t0))
                save_times.append(float(t2 - t1))
                load_times.append(float(t3 - t2))
                peak_rss.append(float((mon.rss_peak - mon.rss0) / (1024**2)))  # delta peak (MB)
                file_sizes.append(float(tmp_path.stat().st_size / (1024**2)))

                # Clean up aggressively
                del data
                gc.collect()
                if tmp_path.exists():
                    tmp_path.unlink()

            rows.append(
                {
                    'x_generator': cfg.x_generator,
                    'y_generator': cfg.y_generator,
                    'P': int(P),
                    'N': int(N),
                    'gen_time_sec_mean': float(np.mean(gen_times)),
                    'gen_time_sec_std': float(np.std(gen_times)),
                    'save_time_sec_mean': float(np.mean(save_times)),
                    'save_time_sec_std': float(np.std(save_times)),
                    'load_time_sec_mean': float(np.mean(load_times)),
                    'load_time_sec_std': float(np.std(load_times)),
                    'peak_rss_mb_delta_mean': float(np.mean(peak_rss)),
                    'peak_rss_mb_delta_std': float(np.std(peak_rss)),
                    'file_size_mb_mean': float(np.mean(file_sizes)),
                    'file_size_mb_std': float(np.std(file_sizes)),
                }
            )

    # Write CSV
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Parallel scaling test on a representative setting
    P0 = 10_000 if 10_000 in p_list else int(p_list[-1])
    N0 = 100 if 100 in n_list else int(n_list[0])

    cfg0 = LargePSmallNSynthConfig(**{**cfg_template.__dict__})
    cfg0.n_features = int(P0)
    cfg0.n_samples = int(N0)
    cfg0.__post_init__()

    scaling_lines: List[str] = []

    # Generation scaling (generate->save)
    base_out = out_dir / "scaling_gen"
    if base_out.exists():
        for p in base_out.glob("*.npz"):
            p.unlink()
    base_out.mkdir(parents=True, exist_ok=True)

    gen_scaling: List[Tuple[int, float, float]] = []  # (jobs, time_sec, ds_per_sec)
    for jobs in gen_jobs_list:
        # Remove previous files
        for p in base_out.glob("*.npz"):
            p.unlink()
        t0 = time.perf_counter()
        _ = generate_many(cfg0, batch_size_for_scaling, n_jobs=int(jobs), base_seed=123, out_dir=base_out, compress=compress, overwrite=True)
        t1 = time.perf_counter()
        dt = float(t1 - t0)
        thr = float(batch_size_for_scaling / max(dt, 1e-12))
        gen_scaling.append((int(jobs), dt, thr))

    # Load scaling
    paths = sorted(str(p) for p in base_out.glob("*.npz"))
    load_scaling: List[Tuple[int, float, float]] = []
    for jobs in gen_jobs_list:
        dt = benchmark_load(paths, n_jobs=int(jobs))
        thr = float(len(paths) / max(dt, 1e-12))
        load_scaling.append((int(jobs), float(dt), float(thr)))

    # Produce a compact markdown report (1 main table + 3 conclusion lines)
    def _md_table(rs: List[Dict[str, Any]]) -> str:
        headers = list(rs[0].keys())
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rs:
            lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
        return "\n".join(lines)

    # Sort rows for readability
    rows_sorted = sorted(rows, key=lambda d: (int(d["P"]), int(d["N"])))
    main_table = _md_table(rows_sorted)

    # Build 3-line conclusions using the representative setting
    # Find the representative row
    rep = None
    for r in rows_sorted:
        if int(r["P"]) == int(P0) and int(r["N"]) == int(N0):
            rep = r
            break
    if rep is None:
        rep = rows_sorted[0]

    best_gen = max(gen_scaling, key=lambda t: t[2])
    best_load = max(load_scaling, key=lambda t: t[2])

    conclusion = [
        f"(1) Representative setting P={P0}, N={N0}: gen_time_mean={rep['gen_time_sec_mean']:.4f}s, save_time_mean={rep['save_time_sec_mean']:.4f}s, load_time_mean={rep['load_time_sec_mean']:.4f}s, peak_rss_delta_mean={rep['peak_rss_mb_delta_mean']:.1f}MB, file_size_mean={rep['file_size_mb_mean']:.1f}MB.",
        f"(2) Generation scaling on {batch_size_for_scaling} datasets: best throughput={best_gen[2]:.2f} ds/s at n_jobs={best_gen[0]} (wall={best_gen[1]:.2f}s).",
        f"(3) Load scaling on {len(paths)} files: best throughput={best_load[2]:.2f} files/s at n_jobs={best_load[0]} (wall={best_load[1]:.2f}s).",
    ]

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Feasibility Report\n\n")
        f.write(f"Config: x_generator={cfg_template.x_generator}, y_generator={cfg_template.y_generator}, compress={compress}\n\n")
        f.write("## Main Results\n\n")
        f.write(main_table)
        f.write("\n\n")
        f.write("## Conclusions (3 lines)\n\n")
        for line in conclusion:
            f.write(f"- {line}\n")

    return report_md



# ---------------------------------
# 8) On-the-fly batch generation (NO save, multi-core)
# ---------------------------------

# Why this exists:
# - For HDD / network storage environments, saving per-iteration batches creates a hard I/O bottleneck.
# - The default training path should therefore be: generate -> train (in-memory), with multi-core CPU
#   generation happening every iteration.
#
# Reproducibility without saving:
# - Deterministically map (step, i) -> seed, and store (seed, step, i) in meta.
# - Then any batch can be reproduced later by re-calling generate_one(seed=...).

_GLOBAL_POOL_GEN: Optional["LargePSmallNSynthGenerator"] = None


def _pool_worker_init(cfg: "LargePSmallNSynthConfig") -> None:
    """Initializer for ProcessPool workers."""
    global _GLOBAL_POOL_GEN
    # Seed here is irrelevant because we always pass an explicit seed into generate_one().
    _GLOBAL_POOL_GEN = LargePSmallNSynthGenerator(cfg, seed=0)


def _pool_generate_one(seed: int, step: int) -> Dict[str, Any]:
    """Worker-side generation function (picklable, top-level)."""
    global _GLOBAL_POOL_GEN
    if _GLOBAL_POOL_GEN is None:
        raise RuntimeError("Pool worker is not initialized. Use initializer=_pool_worker_init.")
    return _GLOBAL_POOL_GEN.generate_one(seed=int(seed), step=int(step), return_graph=False)


def make_batch_seeds(
    base_seed: int,
    step: int,
    batch_size: int,
    seed_stride: int = 100_000,
) -> List[int]:
    """Deterministically map (step, i) -> seed.

    seed(step, i) = base_seed + step * seed_stride + i

    - Pick seed_stride >= batch_size to avoid overlaps.
    - This is the key to "no-save but reproducible" training.

    Returns: List[int] of length batch_size.
    """
    base_seed = int(base_seed)
    step = int(step)
    bs = int(batch_size)
    stride = int(seed_stride)
    if bs < 1:
        raise ValueError("batch_size must be >= 1")
    if stride < bs:
        raise ValueError(f"seed_stride must be >= batch_size (got seed_stride={stride}, batch_size={bs})")
    return [base_seed + step * stride + i for i in range(bs)]


def _annotate_stream_meta(tasks: List[Dict[str, Any]], step: int) -> None:
    """Add stream-level identifiers to meta (in-place)."""
    for bi, t in enumerate(tasks):
        meta = t.get("meta", {})
        meta["stream_step"] = int(step)
        meta["stream_batch_index"] = int(bi)
        # NOTE: meta already contains "seed".
        t["meta"] = meta


def collate_task_list_to_torch(
    tasks: Sequence[Dict[str, Any]],
    *,
    device: Optional[str] = None,
    return_y_hat: bool = False,
    make_p0_mask: bool = False,
) -> Dict[str, Any]:
    """Collate a list of generated tasks into a single torch batch.

    Output:
      - X: (B, N, P) float tensor
      - y: (B, N) tensor (float for regression, long for classification)
      - (optional) y_hat: (B, N) float tensor
      - (optional) p0_mask: (B, P) bool tensor
      - meta: List[dict] (NOT collated)
    """
    if torch is None:
        raise ImportError("collate_task_list_to_torch requires PyTorch (torch).")

    tasks = list(tasks)
    if len(tasks) == 0:
        raise ValueError("Empty task list")

    X = torch.stack([torch.from_numpy(t["X"]) for t in tasks], dim=0)
    y = torch.stack([torch.from_numpy(t["y"]) for t in tasks], dim=0)

    out: Dict[str, Any] = {"X": X, "y": y, "meta": [t.get("meta", {}) for t in tasks]}

    if return_y_hat:
        out["y_hat"] = torch.stack([torch.from_numpy(t["y_hat"]) for t in tasks], dim=0)

    if make_p0_mask:
        P = int(tasks[0]["X"].shape[1])
        mask = torch.zeros((len(tasks), P), dtype=torch.bool)
        for bi, t in enumerate(tasks):
            idx = t.get("meta", {}).get("relevant_feature_indices", np.empty(0, dtype=np.int64))
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            if idx_t.numel():
                mask[bi, idx_t] = True
        out["p0_mask"] = mask

    if device is not None:
        out["X"] = out["X"].to(device)
        out["y"] = out["y"].to(device)
        if "y_hat" in out:
            out["y_hat"] = out["y_hat"].to(device)
        if "p0_mask" in out:
            out["p0_mask"] = out["p0_mask"].to(device)

    return out


class OnTheFlyBatchPool:
    """Multi-process, no-save batch generator for training loops.

    This is designed to be *the default* for training:
      - No disk I/O.
      - Multi-core generation per iteration.
      - Deterministic seed schedule => reproducible without saving.

    Example:
        cfg = LargePSmallNSynthConfig(...)
        with OnTheFlyBatchPool(cfg, batch_size=32, n_workers=8, base_seed=0) as stream:
            for step, batch in enumerate(stream):
                X = batch["X"]   # (B,N,P)
                y = batch["y"]   # (B,N)
                ...
                if step == 1000: break

    Notes:
      - Use mp_start="spawn" for safety (PyTorch-friendly).
      - Avoid nested parallelism: if using n_workers>1, keep cfg.n_jobs_features=1.
    """

    def __init__(
        self,
        cfg: "LargePSmallNSynthConfig",
        *,
        batch_size: int,
        n_workers: int,
        base_seed: int = 0,
        seed_stride: int = 100_000,
        prefetch_batches: int = 2,
        mp_start: str = "spawn",
        return_torch: bool = True,
        torch_device: Optional[str] = None,
        return_y_hat: bool = False,
        make_p0_mask: bool = False,
        max_batches: Optional[int] = None,
    ):
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.n_workers = int(n_workers)
        self.base_seed = int(base_seed)
        self.seed_stride = int(seed_stride)
        self.prefetch_batches = int(max(0, prefetch_batches))
        self.mp_start = str(mp_start)
        self.return_torch = bool(return_torch)
        self.torch_device = torch_device
        self.return_y_hat = bool(return_y_hat)
        self.make_p0_mask = bool(make_p0_mask)
        self.max_batches = None if max_batches is None else int(max_batches)

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")

        self._executor: Optional[ProcessPoolExecutor] = None
        self._next_step: int = 0
        self._prefetched: List[Tuple[int, List[Any]]] = []

    def start(self) -> None:
        if self._executor is not None:
            return
        ctx = mp.get_context(self.mp_start)
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_pool_worker_init,
            initargs=(self.cfg,),
        )
        self._next_step = 0
        self._prefetched = []
        # Initial prefetch
        while len(self._prefetched) < self.prefetch_batches and (self.max_batches is None or self._next_step < self.max_batches):
            self._prefetched.append(self._submit_step(self._next_step))
            self._next_step += 1

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._prefetched = []

    def __enter__(self) -> "OnTheFlyBatchPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _submit_step(self, step: int) -> Tuple[int, List[Any]]:
        if self._executor is None:
            raise RuntimeError("Pool is not started. Call start() or use as a context manager.")
        seeds = make_batch_seeds(self.base_seed, int(step), self.batch_size, seed_stride=self.seed_stride)
        futs = [self._executor.submit(_pool_generate_one, int(s), int(step)) for s in seeds]
        return int(step), futs

    def get_batch(self, step: int) -> Dict[str, Any]:
        """Generate a *specific* batch deterministically (useful for debugging/reproducing)."""
        if self._executor is None:
            self.start()
        assert self._executor is not None
        step_i, futs = self._submit_step(int(step))
        tasks = [f.result() for f in futs]
        _annotate_stream_meta(tasks, step_i)
        if self.return_torch:
            return collate_task_list_to_torch(
                tasks,
                device=self.torch_device,
                return_y_hat=self.return_y_hat,
                make_p0_mask=self.make_p0_mask,
            )
        return {"tasks": tasks, "step": step_i}

    def __iter__(self):
        self.start()
        assert self._executor is not None

        yielded = 0
        while self.max_batches is None or yielded < self.max_batches:
            # Pop one prefetched batch, otherwise generate on-demand.
            if self._prefetched:
                step_i, futs = self._prefetched.pop(0)
            else:
                step_i = self._next_step
                step_i, futs = self._submit_step(step_i)
                self._next_step += 1

            tasks = [f.result() for f in futs]
            _annotate_stream_meta(tasks, int(step_i))

            # Refill prefetch queue
            while len(self._prefetched) < self.prefetch_batches and (self.max_batches is None or self._next_step < self.max_batches):
                self._prefetched.append(self._submit_step(self._next_step))
                self._next_step += 1

            if self.return_torch:
                yield collate_task_list_to_torch(
                    tasks,
                    device=self.torch_device,
                    return_y_hat=self.return_y_hat,
                    make_p0_mask=self.make_p0_mask,
                )
            else:
                yield {"tasks": tasks, "step": int(step_i)}

            yielded += 1


# -----------------------------
# PyTorch DataLoader integration
# -----------------------------

class TorchOnTheFlyTaskDataset(IterableDataset):
    """IterableDataset that generates synthetic tasks *on the fly* in each worker process.

    Recommended usage:
        loader = make_on_the_fly_dataloader(cfg, batch_size=32, num_workers=8, base_seed=0)
        for batch in loader:
            ...

    Reproducibility:
      - Each sample has a deterministic seed:
            seed = base_seed + global_idx * seed_stride
      - We store stream_global_idx + seed in meta, so any single task is reproducible later.
    """

    def __init__(
        self,
        cfg: "LargePSmallNSynthConfig",
        *,
        base_seed: int = 0,
        seed_stride: int = 100_000,
        num_tasks: Optional[int] = None,
        return_y_hat: bool = False,
        make_p0_mask: bool = False,
        batch_size_for_curriculum: int = 1,
    ):
        if torch is None or get_worker_info is None:
            raise ImportError("TorchOnTheFlyTaskDataset requires PyTorch (torch).")
        self.cfg = cfg
        self.base_seed = int(base_seed)
        self.seed_stride = int(seed_stride)
        self.num_tasks = None if num_tasks is None else int(num_tasks)
        self.return_y_hat = bool(return_y_hat)
        self.make_p0_mask = bool(make_p0_mask)
        self.batch_size_for_curriculum = int(max(1, int(batch_size_for_curriculum)))

    def __iter__(self):
        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(wi.id)
            num_workers = int(wi.num_workers)

        g = LargePSmallNSynthGenerator(self.cfg, seed=0)

        local_i = 0
        while True:
            global_idx = worker_id + local_i * num_workers
            if self.num_tasks is not None and global_idx >= self.num_tasks:
                return

            seed = self.base_seed + global_idx * self.seed_stride
            step = int(global_idx // self.batch_size_for_curriculum)
            data = g.generate_one(seed=int(seed), step=step, return_graph=False)

            meta = data.get("meta", {})
            meta["stream_global_idx"] = int(global_idx)
            meta["stream_worker_id"] = int(worker_id)
            data["meta"] = meta

            out: Dict[str, Any] = {
                "X": torch.from_numpy(data["X"]),
                "y": torch.from_numpy(data["y"]),
                "meta": meta,
            }
            if self.return_y_hat:
                out["y_hat"] = torch.from_numpy(data["y_hat"])

            if self.make_p0_mask:
                P = int(data["X"].shape[1])
                mask = torch.zeros(P, dtype=torch.bool)
                idx = torch.as_tensor(meta.get("relevant_feature_indices", np.empty(0, dtype=np.int64)), dtype=torch.long)
                if idx.numel():
                    mask[idx] = True
                out["p0_mask"] = mask

            yield out
            local_i += 1


def collate_synth_tasks(
    batch: Sequence[Dict[str, Any]],
    *,
    return_y_hat: bool = False,
    make_p0_mask: bool = False,
) -> Dict[str, Any]:
    """Custom collate_fn that keeps meta as a list (prevents shape issues)."""
    if torch is None:
        raise ImportError("collate_synth_tasks requires PyTorch (torch).")

    X = torch.stack([b["X"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    out: Dict[str, Any] = {"X": X, "y": y, "meta": [b.get("meta", {}) for b in batch]}

    if return_y_hat:
        if "y_hat" not in batch[0]:
            raise ValueError("return_y_hat=True but y_hat is not present in the batch items.")
        out["y_hat"] = torch.stack([b["y_hat"] for b in batch], dim=0)

    if make_p0_mask:
        if "p0_mask" not in batch[0]:
            raise ValueError("make_p0_mask=True but p0_mask is not present in the batch items.")
        out["p0_mask"] = torch.stack([b["p0_mask"] for b in batch], dim=0)

    return out


def make_on_the_fly_dataloader(
    cfg: "LargePSmallNSynthConfig",
    *,
    batch_size: int,
    num_workers: int,
    base_seed: int = 0,
    seed_stride: int = 100_000,
    num_tasks: Optional[int] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool = False,
    return_y_hat: bool = False,
    make_p0_mask: bool = False,
) -> "DataLoader":
    """Build a PyTorch DataLoader that generates synthetic tasks on-the-fly.

    This is the *recommended default* pipeline for training:
      - No save/load.
      - DataLoader workers do multi-core generation.
      - prefetch_factor overlaps generation with training.

    Important:
      - If num_workers == 0, set prefetch_factor=None (PyTorch requirement).
      - meta is returned as a list (not collated).
    """
    if torch is None or DataLoader is None:
        raise ImportError("make_on_the_fly_dataloader requires PyTorch (torch).")

    dataset = TorchOnTheFlyTaskDataset(
        cfg,
        base_seed=int(base_seed),
        seed_stride=int(seed_stride),
        num_tasks=num_tasks,
        return_y_hat=bool(return_y_hat),
        make_p0_mask=bool(make_p0_mask),
    )

    # PyTorch constraint: prefetch_factor only valid for multiprocessing.
    pf = None if int(num_workers) == 0 else int(prefetch_factor)
    pw = bool(persistent_workers) if int(num_workers) > 0 else False

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        collate_fn=partial(collate_synth_tasks, return_y_hat=bool(return_y_hat), make_p0_mask=bool(make_p0_mask)),
        prefetch_factor=pf,
        persistent_workers=pw,
        pin_memory=bool(pin_memory),
    )

# ---------------------------------
# 9) CLI entrypoints
# ---------------------------------

def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated int list."""
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]



# -----------------------------------------------------------------------------
# QC helpers (sigma / SNR sanity + corr heatmap debugging)
# -----------------------------------------------------------------------------
def qc_sigma_check(
    cfg: LargePSmallNSynthConfig,
    *,
    out_dir: str,
    num: int = 200,
    base_seed: int = 0,
    step: int = 0,
) -> str:
    """Generate `num` tasks and dump sigma/SNR diagnostics (CSV + plots) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    g = LargePSmallNSynthGenerator(cfg, seed=0)

    rows: List[Dict[str, Any]] = []
    for i in range(int(num)):
        seed = int(base_seed) + i
        d = g.generate_one(seed=seed, step=int(step), return_graph=False)
        m = d["meta"]
        rows.append(
            {
                "seed": seed,
                "snr_target": float(m.get("snr_target", np.nan)),
                "snr_empirical": float(m.get("snr_empirical", np.nan)),
                "snr_rel_err": float(m.get("snr_rel_err", np.nan)),
                "eps_sigma_raw": float(m.get("eps_sigma_raw", np.nan)),
                "corr_clean_noisy": float(m.get("corr_clean_noisy", np.nan)),
                "var_y_clean": float(m.get("var_y_clean", np.nan)),
                "var_eps": float(m.get("var_eps", np.nan)),
                "eps_dist": m.get("eps_dist", ""),
                "snr_sampling": m.get("snr_sampling", ""),
                "snr_component": m.get("snr_component", ""),
                "snr_weight_moderate": float(m.get("snr_weight_moderate", np.nan))
                if "snr_weight_moderate" in m
                else np.nan,
                "var_y_clean_was_floored": bool(m.get("var_y_clean_was_floored", False)),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "sigma_check.csv")
    df.to_csv(csv_path, index=False)

    # Summary JSON
    summary = {
        "num": int(df.shape[0]),
        "floored_count": int(df["var_y_clean_was_floored"].sum()),
        "snr_target": df["snr_target"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "snr_empirical": df["snr_empirical"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "snr_rel_err": df["snr_rel_err"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "eps_sigma_raw": df["eps_sigma_raw"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "corr_clean_noisy": df["corr_clean_noisy"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "eps_dist_counts": df["eps_dist"].value_counts().to_dict(),
        "snr_component_counts": df["snr_component"].value_counts().to_dict(),
    }
    json_path = os.path.join(out_dir, "sigma_check_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    import matplotlib.pyplot as plt

    def _hist(col: str, fname: str, *, logx: bool = False) -> None:
        plt.figure()
        x = df[col].dropna().values
        plt.hist(x, bins=50)
        if logx:
            plt.xscale("log")
        plt.title(f"{col} histogram (num={int(df.shape[0])}, step={int(step)})")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    _hist("eps_sigma_raw", "sigma_hist.png", logx=True)
    _hist("snr_target", "snr_target_hist.png", logx=True)
    _hist("corr_clean_noisy", "corr_hist.png", logx=False)
    _hist("snr_rel_err", "snr_relerr_hist.png", logx=True)

    # Target vs empirical scatter
    plt.figure()
    plt.scatter(df["snr_target"], df["snr_empirical"], s=10, alpha=0.6)
    lo = float(np.nanmin(df["snr_target"]))
    hi = float(np.nanmax(df["snr_target"]))
    plt.plot([lo, hi], [lo, hi])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("snr_target")
    plt.ylabel("snr_empirical")
    plt.title("SNR target vs empirical")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_target_vs_empirical.png"), dpi=150)
    plt.close()

    return out_dir


def qc_heatmap_debug(
    cfg: LargePSmallNSynthConfig,
    *,
    out_dir: str,
    seed: int,
    step: int = 0,
    p_plot: int = 64,
    use_first_p: bool = True,
) -> str:
    """Generate one task and dump correlation heatmap + constant/duplicate diagnostics."""
    os.makedirs(out_dir, exist_ok=True)
    g = LargePSmallNSynthGenerator(cfg, seed=0)
    d = g.generate_one(seed=int(seed), step=int(step), return_graph=False)
    X = d["X"]
    P = int(X.shape[1])
    p_plot = int(max(2, min(int(p_plot), P)))

    if use_first_p:
        cols = np.arange(p_plot, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed) + 12345)
        cols = rng.choice(np.arange(P), size=p_plot, replace=False).astype(np.int64, copy=False)
        cols.sort()

    Xs = X[:, cols].astype(np.float64, copy=False)

    # Constant / near-constant detection
    var = np.var(Xs, axis=0)
    const_cols = np.where(var < 1e-12)[0].astype(np.int64)
    # Duplicate detection (exact, after rounding)
    # Note: we round to reduce false negatives due to float32 noise.
    Xr = np.round(Xs, decimals=6)
    # Build hashes per column
    import hashlib

    def _col_hash(v: np.ndarray) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(v).view(np.uint8))
        return h.hexdigest()

    hashes = [_col_hash(Xr[:, j]) for j in range(Xr.shape[1])]
    seen: Dict[str, int] = {}
    dups: List[Tuple[int, int]] = []
    for j, h in enumerate(hashes):
        if h in seen:
            dups.append((seen[h], j))
        else:
            seen[h] = j

    # Correlation matrix (handle NaNs)
    C = np.corrcoef(Xs, rowvar=False)
    diag = np.diag(C)
    diag_bad = np.where(~np.isfinite(diag) | (np.abs(diag - 1.0) > 1e-3))[0].astype(np.int64)

    # Save heatmap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(C, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(f"corr heatmap (seed={int(seed)}, step={int(step)}, p={p_plot})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"corr_heatmap_seed{int(seed)}.png"), dpi=200)
    plt.close()

    # Save diagnostics
    diag_path = os.path.join(out_dir, f"corr_heatmap_seed{int(seed)}_diag.json")
    diag_obj = {
        "seed": int(seed),
        "step": int(step),
        "p_plot": int(p_plot),
        "cols": cols.tolist(),
        "const_cols_local": const_cols.tolist(),  # local indices in [0,p_plot)
        "const_cols_global": cols[const_cols].tolist(),
        "diag_bad_local": diag_bad.tolist(),
        "diag_bad_global": cols[diag_bad].tolist(),
        "num_duplicates": int(len(dups)),
        "duplicate_pairs_local": [(int(a), int(b)) for a, b in dups[:50]],
        "duplicate_pairs_global": [(int(cols[a]), int(cols[b])) for a, b in dups[:50]],
        "x_scale_meta": {k: v for k, v in d["meta"].items() if str(k).startswith("x_scale") or k in ("x_scaling",)},
    }
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_obj, f, indent=2)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic generator (SCM/BNN/Gaussian) + feasibility runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    p_gen = sub.add_parser("generate", help="Generate datasets and (optionally) save to disk")
    p_gen.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p_gen.add_argument("--num", type=int, default=10, help="Number of datasets")
    p_gen.add_argument("--n_jobs", type=int, default=-1, help="Joblib n_jobs")
    p_gen.add_argument("--base_seed", type=int, default=0, help="Base seed")
    p_gen.add_argument("--compress", action="store_true", help="Use np.savez_compressed")
    p_gen.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # feasibility
    p_f = sub.add_parser("feasibility", help="Run feasibility tests and write a markdown report")
    p_f.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p_f.add_argument("--p_list", type=str, default="1000,5000,10000,20000", help="Comma-separated P list")
    p_f.add_argument("--n_list", type=str, default="50,100,500,1000", help="Comma-separated N list")
    p_f.add_argument("--repeats", type=int, default=3, help="Repeats per (P,N)")
    p_f.add_argument("--gen_jobs", type=str, default="1,2,4,8", help="Comma-separated n_jobs list for scaling")
    p_f.add_argument("--batch", type=int, default=32, help="Batch size for scaling")
    p_f.add_argument("--compress", action="store_true", help="Use np.savez_compressed")

    # on-the-fly demo (NO save, DataLoader pipeline)
    p_o = sub.add_parser("onfly_demo", help="Benchmark on-the-fly batch generation (no disk I/O)")
    p_o.add_argument("--steps", type=int, default=50, help="Number of iterations/batches to iterate")
    p_o.add_argument("--batch_size", type=int, default=32, help="Batch size (datasets per iteration)")
    p_o.add_argument("--num_workers", type=int, default=8, help="DataLoader num_workers (multi-core generation)")
    p_o.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor (num_workers>0)")
    p_o.add_argument("--base_seed", type=int, default=0, help="Base seed for reproducibility")
    p_o.add_argument("--seed_stride", type=int, default=100_000, help="Seed stride for step/index mapping")
    p_o.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory")
    p_o.add_argument("--return_y_hat", action="store_true", help="Include y_hat in the batch")
    p_o.add_argument("--make_p0_mask", action="store_true", help="Include p0_mask (B,P) in the batch")

    # QC: sigma / SNR sanity check
    p_s = sub.add_parser("sigma_check", help="QC: check sigma/SNR randomness (CSV + plots)")
    p_s.add_argument("--out_dir", type=str, required=True, help="Output directory for CSV/plots")
    p_s.add_argument("--num", type=int, default=200, help="Number of seeds/tasks to sample")
    p_s.add_argument("--base_seed", type=int, default=0, help="Base seed (tasks use base_seed + i)")
    p_s.add_argument("--step", type=int, default=0, help="Curriculum step to use when snr_sampling is curriculum-based")

    # QC: corr heatmap debug (constant/duplicate columns, diagonal issues)
    p_h = sub.add_parser("heatmap_debug", help="QC: debug corr heatmap artifacts for one seed")
    p_h.add_argument("--out_dir", type=str, required=True, help="Output directory for heatmap + diagnostics JSON")
    p_h.add_argument("--seed", type=int, required=True, help="Dataset seed to generate")
    p_h.add_argument("--step", type=int, default=0, help="Curriculum step for SNR sampling")
    p_h.add_argument("--p_plot", type=int, default=64, help="Number of features to include in the heatmap")
    p_h.add_argument("--random_cols", action="store_true", help="Use random feature columns instead of the first p_plot")


    # shared config knobs (minimal set exposed) (minimal set exposed)
    for sp_ in (p_gen, p_f, p_o, p_s, p_h):
        sp_.add_argument("--x_generator", type=str, default="scm", choices=["scm", "bnn", "gaussian"], help="X generator")
        sp_.add_argument("--y_generator", type=str, default="linear_sparse", choices=["hidden", "linear_sparse", "rf_like", "gbdt_like"], help="y generator")
        sp_.add_argument("--task_type", type=str, default="regression", choices=["classification", "regression"], help="Task type")
        sp_.add_argument("--P", type=int, default=10_000, help="Number of features")
        sp_.add_argument("--N", type=int, default=256, help="Number of samples")
        sp_.add_argument("--n_train", type=int, default=None, help="Train/context sample count (for scaling stats)")
        sp_.add_argument("--x_scaling", type=str, default="robust", choices=["standard", "robust", "none"], help="X scaling method")
        sp_.add_argument("--y_scaling", type=str, default="robust", choices=["standard", "robust", "none"], help="y scaling method (regression only)")
        sp_.add_argument("--snr_sampling", type=str, default="mixture_curriculum", choices=["uniform", "log_uniform", "mixture_curriculum"], help="SNR sampling mode")
        sp_.add_argument("--snr_range", type=float, nargs=2, default=(0.5, 10.0), help="SNR range for uniform/log_uniform")
        sp_.add_argument("--snr_low_range", type=float, nargs=2, default=(0.2, 2.0), help="Low-SNR component range")
        sp_.add_argument("--snr_moderate_range", type=float, nargs=2, default=(1.0, 8.0), help="Moderate-SNR component range")
        sp_.add_argument("--snr_w0", type=float, default=0.8, help="Mixture weight for moderate SNR at step=0")
        sp_.add_argument("--snr_w1", type=float, default=0.2, help="Mixture weight for moderate SNR at step=end")
        sp_.add_argument("--snr_curriculum_steps", type=int, default=50000, help="Anneal steps for SNR curriculum")
        sp_.add_argument("--snr_curriculum_power", type=float, default=1.0, help="Anneal power for SNR curriculum")
        sp_.add_argument("--snr_mixture_uniform", action="store_true", help="Sample SNR components uniformly (default: log-uniform)")
        sp_.add_argument("--max_relevant", type=int, default=100, help="Hard cap for relevant features")
        sp_.add_argument("--max_signal_pool", type=int, default=100, help="Hard cap for signal feature pool")

    args = parser.parse_args()

    cfg = LargePSmallNSynthConfig(
        n_features=int(args.P),
        n_samples=int(args.N),
        n_train=(None if args.n_train is None else int(args.n_train)),
        x_generator=str(args.x_generator),
        y_generator=str(args.y_generator),
        task_type=str(args.task_type),
        max_relevant_features=int(args.max_relevant),
        max_signal_pool_features=int(args.max_signal_pool),
        x_scaling=str(args.x_scaling),
        y_scaling=str(args.y_scaling),
        snr_sampling=str(args.snr_sampling),
        snr_range=(float(args.snr_range[0]), float(args.snr_range[1])),
        snr_low_range=(float(args.snr_low_range[0]), float(args.snr_low_range[1])),
        snr_moderate_range=(float(args.snr_moderate_range[0]), float(args.snr_moderate_range[1])),
        snr_mixture_w_moderate_start=float(args.snr_w0),
        snr_mixture_w_moderate_end=float(args.snr_w1),
        snr_curriculum_steps=int(args.snr_curriculum_steps),
        snr_curriculum_power=float(args.snr_curriculum_power),
        snr_mixture_log_uniform=(not bool(args.snr_mixture_uniform)),
    )

    if args.cmd == "onfly_demo":
        loader = make_on_the_fly_dataloader(
            cfg,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            base_seed=int(args.base_seed),
            seed_stride=int(args.seed_stride),
            num_tasks=None,
            prefetch_factor=int(args.prefetch_factor),
            persistent_workers=True,
            pin_memory=bool(args.pin_memory),
            return_y_hat=bool(args.return_y_hat),
            make_p0_mask=bool(args.make_p0_mask),
        )
        t0 = time.perf_counter()
        n_batches = 0
        n_tasks = 0
        for i, batch in enumerate(loader):
            # Touch tensors to ensure they are materialized
            _ = int(batch["X"].numel())
            n_batches += 1
            n_tasks += int(batch["X"].shape[0])
            if i + 1 >= int(args.steps):
                break
        dt = float(time.perf_counter() - t0)
        print(f"[onfly_demo] batches={n_batches}, tasks={n_tasks}, wall={dt:.3f}s, tasks/s={n_tasks / max(dt, 1e-12):.2f}")
        return

    if args.cmd == "generate":
        paths = generate_many(
            cfg,
            num_datasets=int(args.num),
            n_jobs=int(args.n_jobs),
            base_seed=int(args.base_seed),
            out_dir=str(args.out_dir),
            compress=bool(args.compress),
            overwrite=bool(args.overwrite),
        )
        print(f"Saved {len(paths)} datasets to: {args.out_dir}")
        print("Example:", paths[0] if paths else "(none)")
        return

    if args.cmd == "feasibility":
        p_list = _parse_int_list(str(args.p_list))
        n_list = _parse_int_list(str(args.n_list))
        gen_jobs_list = _parse_int_list(str(args.gen_jobs))
        report = run_feasibility(
            cfg_template=cfg,
            out_dir=str(args.out_dir),
            p_list=p_list,
            n_list=n_list,
            repeats=int(args.repeats),
            gen_jobs_list=gen_jobs_list,
            batch_size_for_scaling=int(args.batch),
            compress=bool(args.compress),
        )
        print(f"Wrote report: {report}")
        return

    if args.cmd == "sigma_check":
        qc_sigma_check(
            cfg,
            out_dir=str(args.out_dir),
            num=int(args.num),
            base_seed=int(args.base_seed),
            step=int(args.step),
        )
        print(f"[sigma_check] wrote reports to: {str(args.out_dir)}")
        return

    if args.cmd == "heatmap_debug":
        qc_heatmap_debug(
            cfg,
            out_dir=str(args.out_dir),
            seed=int(args.seed),
            step=int(args.step),
            p_plot=int(args.p_plot),
            use_first_p=(not bool(args.random_cols)),
        )
        print(f"[heatmap_debug] wrote reports to: {str(args.out_dir)}")
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
