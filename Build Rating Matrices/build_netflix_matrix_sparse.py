#!/usr/bin/env python3
"""
Build a sparse user x movie rating matrix from the Kaggle Netflix Prize combined_data_*.txt files.

Input format (per combined_data_k.txt):
    MovieID:
    UserID,Rating,Date
    UserID,Rating,Date
    ...

Outputs:
    netflix_ratings_csr.npz  : scipy.sparse CSR matrix of shape (num_users, num_movies)
    user_index.csv           : maps row index -> user_id
    movie_index.csv          : maps col index -> movie_id

Rows  = users (raters)
Cols  = movies (items)
Value = rating (1-5)
"""

import os
import csv
import numpy as np
from scipy.sparse import coo_matrix, save_npz

# ========= CONFIGURE THESE =========
DATA_DIR = "."          # directory containing combined_data_1.txt, ..., combined_data_4.txt
OUT_MATRIX = "netflix_ratings_csr.npz"
OUT_USER_INDEX = "user_index.csv"
OUT_MOVIE_INDEX = "movie_index.csv"
# ===================================


def iter_combined_data(data_dir):
    """
    Iterate over all combined_data_*.txt files in data_dir and yield (user_id, movie_id, rating).

    Expects format:
        MovieID:
        UserID,Rating,Date
    """
    # Pick up files named like combined_data_1.txt, combined_data_2.txt, etc.
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().startswith("combined_data_") and f.lower().endswith(".txt")
    )

    if not files:
        raise FileNotFoundError(
            f"No combined_data_*.txt files found in {data_dir}. "
            "Make sure you've extracted them into this directory."
        )

    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"Reading {path} ...")

        with open(path, "r", encoding="latin-1") as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Movie header line: e.g. "1:"
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                    continue

                # Rating line: "UserID,Rating,Date"
                parts = line.split(",")
                if movie_id is None or len(parts) < 2:
                    continue

                user_id = int(parts[0])
                rating = int(parts[1])

                yield user_id, movie_id, rating


def collect_unique_ids(data_dir):
    """
    One pass through combined_data files to collect all distinct user_ids and movie_ids.
    """
    users = set()
    movies = set()

    for u, m, _ in iter_combined_data(data_dir):
        users.add(u)
        movies.add(m)

    users = sorted(users)
    movies = sorted(movies)
    return users, movies


def build_sparse_matrix(data_dir, users, movies):
    """
    Build a COO sparse matrix from Netflix ratings, then convert to CSR.

    Returns:
        csr_matrix of shape (len(users), len(movies))
    """
    from scipy.sparse import coo_matrix  # local import just in case

    user_to_row = {u: i for i, u in enumerate(users)}
    movie_to_col = {m: j for j, m in enumerate(movies)}

    row_idx = []
    col_idx = []
    data = []

    for u, m, r in iter_combined_data(data_dir):
        row_idx.append(user_to_row[u])
        col_idx.append(movie_to_col[m])
        data.append(r)

    row_idx = np.array(row_idx, dtype=np.int32)
    col_idx = np.array(col_idx, dtype=np.int32)
    data = np.array(data, dtype=np.float32)

    coo = coo_matrix((data, (row_idx, col_idx)),
                     shape=(len(users), len(movies)))

    # In case of duplicates (user,movie), COO->CSR sums them,
    # but Netflix data should have at most one rating per pair.
    csr = coo.tocsr()
    return csr


def write_index_csv(ids, path, header_name):
    """
    Write a CSV mapping:
        index,<header_name>
    where 'index' is the row/column index in the matrix
    and <header_name> is user_id or movie_id.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", header_name])
        for idx, _id in enumerate(ids):
            w.writerow([idx, _id])


def main():
    print("Collecting unique user and movie IDs...")
    users, movies = collect_unique_ids(DATA_DIR)
    print(f"Found {len(users)} unique users, {len(movies)} unique movies.")

    print("Building sparse rating matrix...")
    csr = build_sparse_matrix(DATA_DIR, users, movies)
    print(f"Matrix shape: {csr.shape}")
    print(f"Non-zero entries: {csr.nnz}")

    print(f"Saving sparse matrix to {OUT_MATRIX} ...")
    save_npz(OUT_MATRIX, csr)

    print(f"Saving user index to {OUT_USER_INDEX} ...")
    write_index_csv(users, OUT_USER_INDEX, "user_id")

    print(f"Saving movie index to {OUT_MOVIE_INDEX} ...")
    write_index_csv(movies, OUT_MOVIE_INDEX, "movie_id")

    print("Done.")


if __name__ == "__main__":
    main()
