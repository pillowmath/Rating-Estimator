#!/usr/bin/env python3
"""
Build a sparse user x restaurant rating matrix from the Yelp Open Dataset,
then filter to restaurants with >= MIN_RATINGS_PER_RESTAURANT ratings and
users with >= MIN_RATINGS_PER_USER ratings.

Outputs:
    yelp_ratings_csr.npz   : scipy.sparse CSR matrix of shape
                             (num_users, num_restaurants)
    user_index.csv         : maps row index -> user_id
    business_index.csv     : maps col index -> business_id (restaurants)
    yelp_rating_matrix.csv : OPTIONAL dense CSV with:
                                first column = user_id
                                other columns = restaurant business_ids
"""

import os
import csv
import json
import numpy as np
from scipy.sparse import coo_matrix, save_npz

# ========= CONFIGURE THESE =========
DATA_DIR = "."  # directory containing Yelp JSON files

BUSINESS_JSON = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
REVIEW_JSON   = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")

OUT_MATRIX_SPARSE   = "yelp_ratings_csr.npz"
OUT_USER_INDEX      = "user_index.csv"
OUT_BUSINESS_INDEX  = "business_index.csv"

# Optional: also write a dense CSV matrix with user_id as first column
WRITE_DENSE_MATRIX_CSV = False
OUT_MATRIX_DENSE_CSV   = "yelp_rating_matrix.csv"

# Restaurant detection heuristic: look for this substring in categories
RESTAURANT_KEYWORD = "Restaurant"

# If True, only keep businesses with is_open == 1
ONLY_OPEN_BUSINESSES = True

# <<< NEW: minimum rating thresholds >>>
MIN_RATINGS_PER_RESTAURANT = 15
MIN_RATINGS_PER_USER       = 15
# ===================================


def iter_json_lines(path):
    """Yield parsed JSON objects from a newline-delimited JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_restaurant_business_ids():
    """
    Scan business.json to collect business_ids that look like restaurants.

    Returns:
        sorted list of restaurant business_ids
    """
    restaurant_ids = []

    print(f"Scanning businesses in {BUSINESS_JSON} to find restaurants...")

    for obj in iter_json_lines(BUSINESS_JSON):
        bid = obj.get("business_id")
        if not bid:
            continue

        if ONLY_OPEN_BUSINESSES:
            is_open = obj.get("is_open", 1)
            if not is_open:
                continue

        cats = obj.get("categories")

        is_restaurant = False
        if isinstance(cats, list):
            for c in cats:
                if c and RESTAURANT_KEYWORD in c:
                    is_restaurant = True
                    break
        elif isinstance(cats, str):
            if RESTAURANT_KEYWORD in cats:
                is_restaurant = True

        if is_restaurant:
            restaurant_ids.append(bid)

    restaurant_ids = sorted(set(restaurant_ids))
    print(f"Found {len(restaurant_ids)} restaurant businesses.")
    return restaurant_ids


def build_sparse_matrix(restaurant_ids):
    """
    Build a sparse matrix from Yelp reviews restricted to restaurant_ids.

    For each (user, restaurant) pair, if there are multiple reviews,
    we store the **average** star rating.

    Returns:
        csr_matrix, list_of_user_ids, list_of_restaurant_ids
    """
    from scipy.sparse import coo_matrix  # local import

    print(f"Building rating matrix from {REVIEW_JSON} ...")

    # Map business_id -> column index
    biz_to_col = {bid: j for j, bid in enumerate(restaurant_ids)}

    user_to_row = {}
    row_to_user = []  # index -> user_id

    row_idx = []
    col_idx = []
    data = []

    n_lines = 0
    n_used = 0

    for obj in iter_json_lines(REVIEW_JSON):
        n_lines += 1
        bid = obj.get("business_id")
        if bid not in biz_to_col:
            continue  # not a restaurant review

        user_id = obj.get("user_id")
        if user_id is None:
            continue

        stars = obj.get("stars")
        if stars is None:
            continue

        # Assign row index to user on the fly
        if user_id not in user_to_row:
            user_to_row[user_id] = len(row_to_user)
            row_to_user.append(user_id)

        i = user_to_row[user_id]
        j = biz_to_col[bid]

        row_idx.append(i)
        col_idx.append(j)
        data.append(float(stars))
        n_used += 1

        if n_lines % 1_000_000 == 0:
            print(f"  processed {n_lines:,} review lines, {n_used:,} restaurant ratings so far...")

    print(f"Finished reading reviews: {n_lines:,} lines, {n_used:,} restaurant ratings used.")
    print(f"Unique users with restaurant reviews: {len(row_to_user)}")

    # Convert to numpy arrays
    row_idx = np.array(row_idx, dtype=np.int32)
    col_idx = np.array(col_idx, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    num_users = len(row_to_user)
    num_items = len(restaurant_ids)

    # --- 1. Sum of ratings per (user, item) ---
    coo_sum = coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(num_users, num_items)
    )

    # --- 2. Count of ratings per (user, item) ---
    ones = np.ones_like(data, dtype=np.float32)
    coo_count = coo_matrix(
        (ones, (row_idx, col_idx)),
        shape=(num_users, num_items)
    )

    csr_sum = coo_sum.tocsr()
    csr_count = coo_count.tocsr()

    # Divide sum by count entrywise to get average rating
    csr_sum.data = csr_sum.data / csr_count.data
    csr = csr_sum

    print(f"Initial CSR matrix shape: {csr.shape}")
    print(f"Initial non-zero entries: {csr.nnz}")
    print(f"Max rating in matrix: {csr.data.max():.3f}")

    return csr, row_to_user, restaurant_ids


# <<< NEW: filtering function >>>
def filter_by_min_counts(csr, user_ids, restaurant_ids):
    """
    First drop restaurants (columns) with < MIN_RATINGS_PER_RESTAURANT ratings.
    Then drop users (rows) with < MIN_RATINGS_PER_USER ratings.

    Returns:
        filtered_csr, filtered_user_ids, filtered_restaurant_ids
    """
    import numpy as np

    # --- 1. Filter restaurants (columns) ---
    col_counts = np.asarray(csr.getnnz(axis=0)).ravel()
    keep_cols = col_counts >= MIN_RATINGS_PER_RESTAURANT

    num_items_before = csr.shape[1]
    num_items_after = int(keep_cols.sum())
    print(f"Filtering restaurants: {num_items_before} -> {num_items_after} "
          f"(min {MIN_RATINGS_PER_RESTAURANT} ratings)")

    if num_items_after == 0:
        raise RuntimeError("All restaurants were filtered out; try lowering MIN_RATINGS_PER_RESTAURANT.")

    csr = csr[:, keep_cols]
    restaurant_ids = [bid for bid, keep in zip(restaurant_ids, keep_cols) if keep]

    # --- 2. Filter users (rows) ---
    row_counts = np.asarray(csr.getnnz(axis=1)).ravel()
    keep_rows = row_counts >= MIN_RATINGS_PER_USER

    num_users_before = csr.shape[0]
    num_users_after = int(keep_rows.sum())
    print(f"Filtering users: {num_users_before} -> {num_users_after} "
          f"(min {MIN_RATINGS_PER_USER} ratings)")

    if num_users_after == 0:
        raise RuntimeError("All users were filtered out; try lowering MIN_RATINGS_PER_USER.")

    csr = csr[keep_rows, :]
    
    keep_cols = np.asarray(csr.getnnz(axis=0)).ravel() > 0
    csr = csr[:, keep_cols]
    restaurant_ids = [bid for bid, keep in zip(restaurant_ids, keep_cols) if keep]
    
    user_ids = [uid for uid, keep in zip(user_ids, keep_rows) if keep]

    print(f"Filtered CSR matrix shape: {csr.shape}")
    print(f"Filtered non-zero entries: {csr.nnz}")

    return csr, user_ids, restaurant_ids
# <<< END NEW >>>


def write_index_csv(ids, path, header_name):
    """
    Write a CSV mapping:
        index,<header_name>
    where 'index' is the row/column index in the matrix
    and <header_name> is user_id or business_id.
    """
    print(f"Writing index CSV: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", header_name])
        for idx, _id in enumerate(ids):
            w.writerow([idx, _id])


def write_dense_matrix_csv(csr, users, businesses, path):
    """
    Write a dense CSV rating matrix:

        user_id, <business_id_0>, <business_id_1>, ...

    WARNING: This is O(num_users * num_businesses) entries and can be huge.
    """
    num_users, num_items = csr.shape
    print(f"Writing dense rating matrix to {path} ...")
    print(f"  This will write {num_users} x {num_items} cells.")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Header row
        header = ["user_id"] + list(businesses)
        w.writerow(header)

        for i, user_id in enumerate(users):
            dense_row = csr.getrow(i).toarray().ravel()
            row_vals = []
            for val in dense_row:
                if val == 0:
                    row_vals.append("")
                else:
                    row_vals.append(str(val))

            w.writerow([user_id] + row_vals)

            if (i + 1) % 1000 == 0:
                print(f"  wrote {i+1:,} / {num_users:,} users...", end="\r")

    print("\nFinished writing dense CSV.")


def main():
    # 1. Collect restaurant business_ids
    restaurant_ids = collect_restaurant_business_ids()
    if not restaurant_ids:
        raise RuntimeError("No restaurant businesses found. "
                           "Check RESTAURANT_KEYWORD or input paths.")

    # 2. Build sparse rating matrix from reviews
    csr, user_ids, restaurant_ids = build_sparse_matrix(restaurant_ids)

    # 3. Apply min-rating filters (restaurants, then users)
    csr, user_ids, restaurant_ids = filter_by_min_counts(csr, user_ids, restaurant_ids)

    # 4. Save sparse matrix
    print(f"Saving sparse matrix to {OUT_MATRIX_SPARSE} ...")
    save_npz(OUT_MATRIX_SPARSE, csr)

    # 5. Save index CSVs (aligned with filtered matrix)
    write_index_csv(user_ids, OUT_USER_INDEX, "user_id")
    write_index_csv(restaurant_ids, OUT_BUSINESS_INDEX, "business_id")

    # 6. (Optional) Save dense matrix CSV with user_id as first column
    if WRITE_DENSE_MATRIX_CSV:
        write_dense_matrix_csv(csr, user_ids, restaurant_ids, OUT_MATRIX_DENSE_CSV)

    print("Done.")


if __name__ == "__main__":
    main()
