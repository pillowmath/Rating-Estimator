#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ---- Display options for nicer printing ----
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.precision", 6)


# ============================================================
# Quantile helpers
# ============================================================

def rcq_sorted(sorted_samples: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    Right-continuous inverse empirical CDF on a sorted sample.

    Parameters
    ----------
    sorted_samples : 1D np.ndarray
        Sorted sample values (ascending).
    qs : 1D np.ndarray
        Quantiles in [0, 1].

    Returns
    -------
    1D np.ndarray
        Values F^{-1}(q) for each q in qs, using right-continuous indexing.
    """
    n = sorted_samples.size
    idx = np.ceil(qs * n).astype(int) - 1
    idx = np.clip(idx, 0, n - 1)
    return sorted_samples[idx]


def cdf_quantiles_from_scores(scores: pd.Series) -> pd.Series:
    """
    Map item scores (e.g., means or primitive ratings) to right-continuous
    empirical CDF values across items.

    Ties use 'max' rank so they get the highest possible CDF.

    Returns
    -------
    pd.Series
        Same index as 'scores', values in (0, 1].
    """
    n = scores.notna().sum()
    ranks = scores.rank(method="max")  # highest rank among ties
    return ranks / n


# ============================================================
# Methods A, B, C
# ============================================================

def method_A_mean(train_df: pd.DataFrame) -> pd.Series:
    """
    Method A: item-wise mean rating over training raters.
    """
    return train_df.mean(axis=0, skipna=True)


def compute_barycenter_quantiles(train_df: pd.DataFrame):
    """
    Compute W2-style barycenter of users' personal rating distributions.

    The barycenter is represented on a fixed grid of K = 2 * (#items)
    quantile points.

    Returns
    -------
    barycenter : np.ndarray, shape (K,)
        Average quantile curve over users.
    user_sorted_ratings : pd.Series
        Index = user IDs, value = sorted np.ndarray of ratings for that user.
    K : int
        Number of quantile grid points.
    """
    # Quantile grid length
    K = 2 * train_df.shape[1]

    # For each user (row), get sorted non-missing ratings
    user_sorted_ratings = train_df.apply(
        lambda row: np.sort(row.dropna().to_numpy()), axis=1
    )

    # Filter out users with no ratings
    user_arrays = [arr for arr in user_sorted_ratings if arr.size > 0]
    if not user_arrays:
        return np.full(K, np.nan), user_sorted_ratings, K

    # For each user, sample their empirical quantile curve on K grid points
    grid = np.arange(K)
    quantile_rows = []
    for arr in user_arrays:
        L = len(arr)
        # indices along the sorted array corresponding to these K quantiles
        idx = (grid * L) // K
        idx = np.clip(idx, 0, L - 1)
        quantile_rows.append(arr[idx])

    quantile_array = np.vstack(quantile_rows)
    barycenter = quantile_array.mean(axis=0)
    return barycenter, user_sorted_ratings, K


def primitive_rating_scores(train_df: pd.DataFrame) -> pd.Series:
    """
    Method B: primitive rating.

    For each item j:
      1. For each user i who rated j, map their rating r_ij through
         F_{hat mu}^{-1}(F_i(r_ij)), where F_i is user i's empirical CDF
         and hat mu is the W2-barycenter across users.
      2. Average these mapped values over users.

    Returns
    -------
    pd.Series
        Index = item IDs, value = primitive rating for that item.
    """
    barycenter, user_sorted_ratings, K = compute_barycenter_quantiles(train_df)
    items = train_df.columns
    primitive = pd.Series(index=items, dtype=float)

    for item in items:
        col = train_df[item].dropna()
        if col.empty:
            primitive[item] = np.nan
            continue

        user_ids = col.index.to_numpy()
        observed_ratings = col.to_numpy()
        mapped_values = np.empty_like(observed_ratings, dtype=float)

        for i, (user_id, rating) in enumerate(zip(user_ids, observed_ratings)):
            # sorted ratings for this user
            a_u = user_sorted_ratings.loc[user_id]
            if len(a_u) == 0:
                mapped_values[i] = np.nan
                continue

            # right-continuous CDF F_u(r)
            p = np.searchsorted(a_u, rating, side="right") / len(a_u)

            # map p to barycenter quantile F_{hat mu}^{-1}(p)
            j = int(np.ceil(p * K)) - 1
            j = max(0, min(j, K - 1))  # clamp to [0, K-1]
            mapped_values[i] = barycenter[j]

        primitive[item] = np.nanmean(mapped_values)

    return primitive


def method_C_random_rank(train_df: pd.DataFrame, seed: int) -> pd.Series:
    """
    Method C: random ranking of items.

    Assigns strictly increasing scores in a random order; only the ordering
    matters, not the actual values.
    """
    cols = train_df.columns.to_numpy()
    n_items = cols.size
    rng = np.random.default_rng(seed)

    # Base strictly increasing scores, then permute
    base_scores = np.linspace(0.0, 1.0, n_items, endpoint=False)
    perm = rng.permutation(n_items)
    return pd.Series(base_scores[perm], index=cols)


# ============================================================
# Evaluation over users: MAE, MSE, Exact Match
# ============================================================

def eval_quantiles_over_users(user_cache, item_quantiles: pd.Series):
    """
    Evaluate one method (given by item quantiles) over all users in a fold.

    user_cache: dict[user_id] -> (hidden_items, vis_vals_sorted, truth_hidden)
      - hidden_items: 1D array of item IDs whose ratings are hidden
      - vis_vals_sorted: sorted ratings for the *visible* items for that user
      - truth_hidden: true ratings for the hidden_items (same order as hidden)

    item_quantiles: pd.Series
      CDF_{method}(item) for each item (right-continuous across items).

    Returns
    -------
    n_pred : int
        Total number of predictions.
    mae_sum, mae_sumsq : float
        Sum of |error| and sum of |error|^2 across predictions.
    mse_sum, mse_sumsq : float
        Sum of squared error and sum of squared error^2 across predictions.
    em_sum : float
        Sum of exact matches across predictions.
        (We’ll derive its SE using Bernoulli variance.)
    """
    n_pred = 0
    mae_sum = mae_sumsq = 0.0
    mse_sum = mse_sumsq = 0.0
    em_sum = 0.0

    for _, (hidden_items, vis_sorted, truth_hidden) in user_cache.items():
        # Quantiles for the hidden items under this method
        q_hidden = item_quantiles.loc[hidden_items].to_numpy(np.float64)
        valid = ~np.isnan(q_hidden)
        if not valid.any():
            continue

        q_hidden = q_hidden[valid]
        t = truth_hidden[valid]

        # Predict by mapping quantiles through user's visible rating distribution
        preds = rcq_sorted(vis_sorted, q_hidden)

        diffs = preds - t
        abs_d = np.abs(diffs)
        sq_d = diffs ** 2
        matches = (preds == t).astype(float)

        mae_sum += abs_d.sum()
        mae_sumsq += (abs_d ** 2).sum()

        mse_sum += sq_d.sum()
        mse_sumsq += (sq_d ** 2).sum()

        em_sum += matches.sum()
        n_pred += diffs.size

    return n_pred, mae_sum, mae_sumsq, mse_sum, mse_sumsq, em_sum


# ============================================================
# Cross-validation driver
# ============================================================

def run_cross_validated_evaluation(
    n_splits: int = 5,
    hide_frac: float = 0.2,
    random_state: int = 42,
    random_reps: int = 5,
) -> pd.DataFrame:
    """
    k-fold cross-validation over raters (rows).

    For each fold:
      - Train methods A/B/C on the training raters.
      - For each test rater, hide a fraction of their rated items.
      - Predict hidden ratings from the remaining (visible) ratings and item
        quantiles from each method.
      - Pool errors across *all* predictions from all folds (and all random
        reps for method C).

    Returns
    -------
    pd.DataFrame
        One row per method, with pooled mean + SE for MAE, MSE, ExactMatchRate.
    """
    # Load rating matrix (rows = users, columns = items)
    df = pd.read_csv("rating_matrix.csv", index_col=0).astype(np.float32)

    # Shuffle raters and divide into folds
    rng = np.random.default_rng(random_state)
    user_ids = df.index.to_numpy()
    rng.shuffle(user_ids)
    folds = np.array_split(user_ids, n_splits)

    # Accumulators per method
    accum = {
        "A": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0),
        "B": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0),
        "C": dict(n=0, mae=0.0, mae2=0.0, mse=0.0, mse2=0.0, em=0.0),
    }

    total_users_tested = 0
    item_ids = df.columns.to_numpy()

    for fold_idx, test_idx in enumerate(folds):
        # Train/test split for this fold
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        # --------------------------------------------------------
        # Build user_cache: for each test user, which items are hidden,
        # what is their visible (sorted) rating distribution, and the
        # true ratings for the hidden items.
        # --------------------------------------------------------
        user_cache = {}
        rng_fold = np.random.default_rng(random_state + 1000 + fold_idx)

        for u in test_df.index:
            row = test_df.loc[u]
            rated_mask = ~np.isnan(row.to_numpy())
            rated_items = item_ids[rated_mask]
            n_rated = rated_items.size

            # Number of ratings to hide for this user
            k_hide = int(np.floor(hide_frac * n_rated))
            if n_rated >= 2 and 1 <= k_hide < n_rated:
                hidden_items = rng_fold.choice(rated_items, size=k_hide, replace=False)
                visible_items = np.setdiff1d(rated_items, hidden_items, assume_unique=False)

                vis_vals_sorted = np.sort(row.loc[visible_items].to_numpy(np.float64))
                truth_hidden = row.loc[hidden_items].to_numpy(np.float64)
                user_cache[u] = (hidden_items, vis_vals_sorted, truth_hidden)

        total_users_tested += len(user_cache)

        # --------------------------------------------------------
        # Compute per-item scores on training data (A and B),
        # convert to item-quantiles, then evaluate on this fold.
        # --------------------------------------------------------
        # Method A: mean
        scores_A = method_A_mean(train_df).reindex(df.columns)
        q_A = cdf_quantiles_from_scores(scores_A)

        # Method B: primitive
        scores_B = primitive_rating_scores(train_df).reindex(df.columns)
        q_B = cdf_quantiles_from_scores(scores_B)

        for key, q_method in [("A", q_A), ("B", q_B)]:
            n, mae, mae2, mse, mse2, em = eval_quantiles_over_users(user_cache, q_method)
            acc = accum[key]
            acc["n"]   += n
            acc["mae"] += mae;  acc["mae2"] += mae2
            acc["mse"] += mse;  acc["mse2"] += mse2
            acc["em"]  += em

        # --------------------------------------------------------
        # Method C: random ranking. Repeat random_reps times.
        # --------------------------------------------------------
        for rep in range(random_reps):
            seed = (random_state + 777) + fold_idx * random_reps + rep
            scores_C = method_C_random_rank(train_df, seed=seed).reindex(df.columns)
            q_C = cdf_quantiles_from_scores(scores_C)

            n, mae, mae2, mse, mse2, em = eval_quantiles_over_users(user_cache, q_C)
            acc = accum["C"]
            acc["n"]   += n
            acc["mae"] += mae;  acc["mae2"] += mae2
            acc["mse"] += mse;  acc["mse2"] += mse2
            acc["em"]  += em

    # ------------------------------------------------------------
    # Turn raw sums into mean + SE across predictions.
    # ------------------------------------------------------------
    def finalize(acc):
        n = acc["n"]
        if n == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

        mae_mean = acc["mae"] / n
        mse_mean = acc["mse"] / n
        em_mean  = acc["em"]  / n

        # Second-moment-based variance for MAE and MSE
        mae_var = max(acc["mae2"] / n - mae_mean ** 2, 0.0)
        mse_var = max(acc["mse2"] / n - mse_mean ** 2, 0.0)

        # For ExactMatch (0/1), variance is p(1-p)
        em_var  = em_mean * (1.0 - em_mean)

        mae_se = np.sqrt(mae_var) / np.sqrt(n)
        mse_se = np.sqrt(mse_var) / np.sqrt(n)
        em_se  = np.sqrt(em_var)  / np.sqrt(n)

        return (mae_mean, mae_se, mse_mean, mse_se, em_mean, em_se, int(n))

    res_A = finalize(accum["A"])
    res_B = finalize(accum["B"])
    res_C = finalize(accum["C"])

    summary = pd.DataFrame(
        [
            ("Method A (mean)",)      + res_A + (total_users_tested,),
            ("Method B (primitive)",) + res_B + (total_users_tested,),
            (f"Method C (random rank, R={random_reps})",) + res_C + (total_users_tested,),
        ],
        columns=[
            "Method",
            "MAE_mean", "MAE_se",
            "MSE_mean", "MSE_se",
            "ExactMatchRate_mean", "ExactMatchRate_se",
            "n_predictions",
            "n_users_total",
        ],
    ).set_index("Method")

    return summary


# ============================================================
# Run the evaluation
# ============================================================
if __name__ == "__main__":
    n_splits = 5
    summary_cv = run_cross_validated_evaluation(
        n_splits=n_splits,
        hide_frac=0.2,
        random_state=42,
        random_reps=5,
    )
    print(f"\n=== {n_splits}-Fold Cross-Validated Comparison ===")
    print(summary_cv.to_string(index=True))
