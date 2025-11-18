#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, csc_matrix

# ---- Display: show all columns, no ellipses ----
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 6)

# ============ Right-continuous quantile helpers ============

def rcq_sorted(sorted_samples: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Right-continuous inverse CDF for a pre-sorted 1-D sample array."""
    n = sorted_samples.size
    if n == 0:
        return np.full_like(qs, np.nan, dtype=float)
    idx = np.ceil(qs * n).astype(int) - 1
    idx = np.clip(idx, 0, n - 1)
    return sorted_samples[idx]

def cdf_quantiles_from_scores_array(scores: np.ndarray) -> np.ndarray:
    """
    Right-continuous empirical CDF across items (ties take highest rank),
    implemented on a 1D numpy array with NaNs allowed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    q = np.full_like(scores, np.nan, dtype=np.float64)

    mask = ~np.isnan(scores)
    n = mask.sum()
    if n == 0:
        return q

    vals = scores[mask]
    # stable sort so we can do rank(method="max") logic
    order = np.argsort(vals, kind="mergesort")

    ranks = np.empty(n, dtype=np.int64)
    i = 0
    # rank(method="max"): each tied block gets rank = last index in block + 1
    while i < n:
        j = i + 1
        while j < n and vals[order[j]] == vals[order[i]]:
            j += 1
        ranks[i:j] = j
        i = j

    r_full = np.empty(n, dtype=np.float64)
    r_full[order] = ranks
    q[mask] = r_full / n
    return q

# ============ Methods on sparse matrices ============

def method_A_mean_csr(R_train: csr_matrix) -> np.ndarray:
    """
    Item-wise mean over training raters for a CSR matrix.
    Missing = NaN.
    """
    R_csc = R_train.tocsc()
    sums = np.array(R_csc.sum(axis=0)).ravel().astype(np.float64)
    counts = np.diff(R_csc.indptr).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = sums / counts
        means[counts == 0] = np.nan
    return means

def compute_barycenter_quantiles_csr(R_train: csr_matrix):
    """
    Compute the W2-style barycenter on a K=2*#items grid using users'
    sorted rating arrays from a CSR matrix.

    Returns:
        barycenter : np.ndarray of shape (K,)
        user_arrays: list of per-user sorted rating arrays (or None if empty)
        K          : int
    """
    n_users, n_items = R_train.shape
    K = 2 * n_items
    grid = np.arange(K)

    # First pass: accumulate quantile rows of each non-empty user
    rows = []
    for u in range(n_users):
        start, end = R_train.indptr[u], R_train.indptr[u + 1]
        if start == end:
            continue
        vals = np.sort(R_train.data[start:end])
        L = vals.size
        idx = (grid * L) // K
        idx = np.clip(idx, 0, L - 1)
        rows.append(vals[idx])

    if not rows:
        barycenter = np.full(K, np.nan, dtype=np.float64)
    else:
        quantile_array = np.vstack(rows)
        barycenter = quantile_array.mean(axis=0)

    # Second pass: cache each user's sorted rating array
    user_arrays = []
    for u in range(n_users):
        start, end = R_train.indptr[u], R_train.indptr[u + 1]
        if start == end:
            user_arrays.append(None)
        else:
            user_arrays.append(np.sort(R_train.data[start:end]))

    return barycenter, user_arrays, K

def primitive_rating_scores_csr(R_train: csr_matrix) -> np.ndarray:
    """
    Method B: "primitive rating" scores from CSR matrix.

    For each item j:
      - For each user u who rated j with r_uj, compute p = F_u(r_uj) (right-continuous ECDF),
        map via barycenter grid, and average mapped values.
    """
    n_users, n_items = R_train.shape
    barycenter, user_arrays, K = compute_barycenter_quantiles_csr(R_train)
    R_csc = R_train.tocsc()

    scores = np.full(n_items, np.nan, dtype=np.float64)

    for j in range(n_items):
        cstart, cend = R_csc.indptr[j], R_csc.indptr[j + 1]
        if cstart == cend:
            continue

        rows = R_csc.indices[cstart:cend]
        vals = R_csc.data[cstart:cend]
        mapped = np.empty_like(vals, dtype=np.float64)

        for idx, (u, r) in enumerate(zip(rows, vals)):
            arr = user_arrays[u]
            if arr is None or arr.size == 0:
                mapped[idx] = np.nan
                continue
            L = arr.size
            # right-continuous F_u(r)
            p = np.searchsorted(arr, r, side="right") / L
            jj = int(np.ceil(p * K)) - 1
            if jj < 0:
                jj = 0
            elif jj >= K:
                jj = K - 1
            mapped[idx] = barycenter[jj]

        ok = ~np.isnan(mapped)
        if not np.any(ok):
            continue

        scores[j] = mapped[ok].mean()

    return scores

def method_C_random_rank(n_items: int, seed: int) -> np.ndarray:
    """
    Random ranking baseline: strictly increasing scores in random order.
    Only the order matters; values are arbitrary.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    base = np.linspace(0.0, 1.0, n_items, endpoint=False)
    scores = np.empty(n_items, dtype=np.float64)
    scores[perm] = base
    return scores

# ============ CV evaluation on sparse matrix ============

def eval_quantiles_over_users_sparse(user_cache, q_method: np.ndarray):
    """
    Pooled moments across all predictions (MAE, MSE, ExactMatch).
    user_cache: dict[user_id] -> (hidden_idx, vis_vals_sorted, truth_hidden)
      - hidden_idx: 1D array of item indices for hidden ratings
      - vis_vals_sorted: sorted visible rating values for that user
      - truth_hidden: actual ratings for hidden_idx (same order)
    q_method: length-n_items array of item quantiles in [0,1]
    """
    n_pred = 0
    mae_sum = mse_sum = match_sum = 0.0
    mae_sumsq = mse_sumsq = match_sumsq = 0.0

    for hidden_idx, vis_vals_sorted, truth_hidden in user_cache.values():
        p = q_method[hidden_idx].astype(np.float64)
        ok = ~np.isnan(p)
        if not np.any(ok):
            continue

        p = p[ok]
        t = truth_hidden[ok]

        preds = rcq_sorted(vis_vals_sorted, p)
        diffs = preds - t

        abs_d = np.abs(diffs)
        sq_d = diffs ** 2
        match = (preds == t).astype(float)

        mae_sum += abs_d.sum();     mae_sumsq += (abs_d ** 2).sum()
        mse_sum += sq_d.sum();      mse_sumsq += (sq_d ** 2).sum()
        match_sum += match.sum();   match_sumsq += (match ** 2).sum()
        n_pred += diffs.size

    return n_pred, mae_sum, mae_sumsq, mse_sum, mse_sumsq, match_sum, match_sumsq

def run_cross_validated_evaluation_sparse(
    matrix_path: str = "rating_matrix.npz",
    n_splits: int = 5,
    hide_frac: float = 0.2,
    random_state: int = 42,
    random_reps: int = 5,
    max_users: int = None,  # optionally subsample users for speed
) -> pd.DataFrame:
    """
    k-fold CV over raters (rows) using a sparse CSR matrix.

    - Each fold: train on K-1 folds, test on held-out fold.
    - For each test user, hide a fraction of their rated items,
      predict via rcq/barycenter machinery, pool errors across all folds.
    """
    import math

    R_full = load_npz(matrix_path).astype(np.float32)
    n_users, n_items = R_full.shape

    # Optional: restrict to a subset of users for tractability
    if max_users is not None and max_users < n_users:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(n_users, size=max_users, replace=False)
        chosen.sort()
        R_full = R_full[chosen, :]
        n_users = max_users

    rng = np.random.default_rng(random_state)
    user_indices = np.arange(n_users, dtype=np.int64)
    rng.shuffle(user_indices)
    folds = np.array_split(user_indices, n_splits)

    accum = {
        "A": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0, em2=0.0),
        "B": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0, em2=0.0),
        "C": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0, em2=0.0),
    }
    total_users_tested = 0

    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])

        R_train = R_full[train_idx, :]
        R_test = R_full[test_idx, :]

        # Build per-user cache for this fold
        user_cache = {}
        rng_fold = np.random.default_rng(random_state + 1000 + k)

        for i, u in enumerate(test_idx):
            # R_test is CSR; row i corresponds to global user u
            start, end = R_test.indptr[i], R_test.indptr[i + 1]
            cols = R_test.indices[start:end]
            vals = R_test.data[start:end].astype(np.float64)
            n_rated = cols.size

            if n_rated < 2:
                continue

            k_hide = int(math.floor(hide_frac * n_rated))
            if k_hide < 1 or k_hide >= n_rated:
                continue

            hide_pos = rng_fold.choice(n_rated, size=k_hide, replace=False)
            hidden_idx = cols[hide_pos]
            visible_mask = np.ones(n_rated, dtype=bool)
            visible_mask[hide_pos] = False

            vis_vals_sorted = np.sort(vals[visible_mask])
            truth_hidden = vals[hide_pos]

            if vis_vals_sorted.size == 0:
                continue

            user_cache[int(u)] = (hidden_idx, vis_vals_sorted, truth_hidden)

        total_users_tested += len(user_cache)
        if not user_cache:
            continue

        # ---- Method A (mean) ----
        scores_A = method_A_mean_csr(R_train)
        q_A = cdf_quantiles_from_scores_array(scores_A)
        n, mae, mae2, mse, mse2, em, em2 = eval_quantiles_over_users_sparse(user_cache, q_A)
        acc = accum["A"]
        acc["n"] += n
        acc["mae"] += mae;  acc["mae2"] += mae2
        acc["mse"] += mse;  acc["mse2"] += mse2
        acc["em"]  += em;   acc["em2"]  += em2

        # ---- Method B (primitive) ----
        scores_B = primitive_rating_scores_csr(R_train)
        q_B = cdf_quantiles_from_scores_array(scores_B)
        n, mae, mae2, mse, mse2, em, em2 = eval_quantiles_over_users_sparse(user_cache, q_B)
        acc = accum["B"]
        acc["n"] += n
        acc["mae"] += mae;  acc["mae2"] += mae2
        acc["mse"] += mse;  acc["mse2"] += mse2
        acc["em"]  += em;   acc["em2"]  += em2

        # ---- Method C (random rank baseline) ----
        for r in range(random_reps):
            seed = (random_state + 777) + k * random_reps + r
            scores_C = method_C_random_rank(n_items, seed)
            q_C = cdf_quantiles_from_scores_array(scores_C)
            n, mae, mae2, mse, mse2, em, em2 = eval_quantiles_over_users_sparse(user_cache, q_C)
            acc = accum["C"]
            acc["n"] += n
            acc["mae"] += mae;  acc["mae2"] += mae2
            acc["mse"] += mse;  acc["mse2"] += mse2
            acc["em"]  += em;   acc["em2"]  += em2

    # ---- Convert accumulators to mean + SE ----
    def finalize(acc):
        n = acc["n"]
        if n == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
        mae_mean = acc["mae"] / n
        mse_mean = acc["mse"] / n
        em_mean  = acc["em"]  / n

        mae_var = max(acc["mae2"] / n - mae_mean**2, 0.0)
        mse_var = max(acc["mse2"] / n - mse_mean**2, 0.0)
        em_var  = max(acc["em2"]  / n - em_mean**2, 0.0)

        mae_se = np.sqrt(mae_var) / np.sqrt(n)
        mse_se = np.sqrt(mse_var) / np.sqrt(n)
        em_se  = np.sqrt(em_var)  / np.sqrt(n)

        return (mae_mean, mae_se, mse_mean, mse_se, em_mean, em_se, int(n))

    res_A = finalize(accum["A"])
    res_B = finalize(accum["B"])
    res_C = finalize(accum["C"])

    out = pd.DataFrame(
        [
            ("Method A (mean)",)                       + res_A + (total_users_tested,),
            ("Method B (primitive)",)                  + res_B + (total_users_tested,),
            (f"Method C (random rank, R={random_reps})",) + res_C + (total_users_tested,),
        ],
        columns=[
            "Method",
            "MAE_mean","MAE_se",
            "MSE_mean","MSE_se",
            "ExactMatchRate_mean","ExactMatchRate_se",
            "n_predictions",
            "n_users_total",
        ],
    ).set_index("Method")

    return out

# ---- Run the evaluation ----

n_splits = 5
summary_cv = run_cross_validated_evaluation_sparse(
    matrix_path="rating_matrix.npz",
    n_splits=n_splits,
    hide_frac=0.2,
    random_state=42,
    random_reps=5,
    max_users=None,
)
print(f"\n=== {n_splits}-Fold Cross-Validated Comparison ===")
print(summary_cv.to_string(index=True))
