#!/usr/bin/env python3
"""Convert a user's long-form ratings CSV into this repository's input format.

Only the input layout is standardized; the original rating scores are left unchanged.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz


def parse_args():
    # Inputs: Command-line flags for the CSV, its three column names, output directory,
    #         dense/sparse format, duplicate policy, and minimum observation counts.
    # Outputs: An argparse.Namespace of validated command-line values.
    # Assumptions: The input is a readable CSV containing one row per observed rating.
    """Parse command-line arguments for a user-supplied ratings dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Long-form ratings CSV')
    parser.add_argument('--user-column', default='user_id')
    parser.add_argument('--item-column', default='item_id')
    parser.add_argument('--rating-column', default='rating')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--format', choices=['dense', 'sparse'], default='dense')
    parser.add_argument('--duplicate', choices=['mean', 'last'], default='mean',
                        help='How to collapse repeated user-item ratings')
    parser.add_argument('--min-user-ratings', type=int, default=1)
    parser.add_argument('--min-item-ratings', type=int, default=1)
    args = parser.parse_args()
    if args.min_user_ratings < 1 or args.min_item_ratings < 1:
        parser.error('Minimum rating counts must be positive integers')
    return args


def prepare_ratings(raw, user_col, item_col, rating_col, duplicate, min_user, min_item):
    # Inputs: pandas DataFrame; original user, item and rating field names;
    #         duplicate policy and minimum number of observed entries per row/column.
    # Outputs: DataFrame with the columns user_id, item_id, rating, one row per pair.
    # Assumptions: User/item identifiers may be cast to strings; ratings must be finite
    #              numeric values. Filtering is iterated until both thresholds hold.
    """Validate and de-duplicate long-form ratings, then enforce observation thresholds."""
    cols = [user_col, item_col, rating_col]
    if len(set(cols)) != 3 or any(c not in raw.columns for c in cols):
        raise ValueError(f'Expected three distinct CSV columns: {cols}; found {list(raw.columns)}')
    df = raw[cols].copy()
    df.columns = ['user_id', 'item_id', 'rating']
    df = df.dropna(subset=['user_id', 'item_id', 'rating'])
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='raise')
    if not np.isfinite(df['rating'].to_numpy()).all():
        raise ValueError('All retained rating values must be finite')
    if duplicate == 'mean':
        df = df.groupby(['user_id', 'item_id'], as_index=False, sort=True)['rating'].mean()
    else:
        df = df.drop_duplicates(['user_id', 'item_id'], keep='last')
    while len(df):
        users_ok = df.groupby('user_id')['rating'].transform('size') >= min_user
        items_ok = df.groupby('item_id')['rating'].transform('size') >= min_item
        filtered = df[users_ok & items_ok].copy()
        if len(filtered) == len(df):
            break
        df = filtered
    if df.empty:
        raise ValueError('No ratings remain after validation and minimum-count filtering')
    return df.sort_values(['user_id', 'item_id']).reset_index(drop=True)


def write_outputs(df, output_dir, output_format):
    # Inputs: Validated long-form rating rows, output Path, and 'dense'/'sparse' choice.
    # Outputs: Dense rating_matrix.csv or sparse rating_matrix.npz, plus index CSVs.
    # Assumptions: Sparse format represents missing data by implicit zeros and thus
    #              cannot represent a genuinely observed rating of exactly zero.
    """Save ratings in the exact format expected by the dense or sparse estimator."""
    if output_format == 'sparse' and (df['rating'] == 0).any():
        raise ValueError('Sparse format cannot distinguish observed zero ratings from missing entries; use --format dense')
    output_dir.mkdir(parents=True, exist_ok=True)
    users = sorted(df['user_id'].unique())
    items = sorted(df['item_id'].unique())
    pd.DataFrame({'index': np.arange(len(users)), 'user_id': users}).to_csv(
        output_dir / 'user_index.csv', index=False)
    pd.DataFrame({'index': np.arange(len(items)), 'item_id': items}).to_csv(
        output_dir / 'item_index.csv', index=False)
    if output_format == 'dense':
        dense = df.pivot(index='user_id', columns='item_id', values='rating')
        dense = dense.reindex(index=users, columns=items)
        dense.to_csv(output_dir / 'rating_matrix.csv', index_label='user_id')
        result = output_dir / 'rating_matrix.csv'
    else:
        row = pd.Categorical(df['user_id'], categories=users).codes
        col = pd.Categorical(df['item_id'], categories=items).codes
        matrix = coo_matrix((df['rating'].astype(float).to_numpy(), (row, col)),
                            shape=(len(users), len(items))).tocsr()
        save_npz(output_dir / 'rating_matrix.npz', matrix)
        result = output_dir / 'rating_matrix.npz'
    print(f'Wrote {result} with {len(users)} users, {len(items)} items, {len(df)} observations')
    print(f'ID mappings: {output_dir / "user_index.csv"} and {output_dir / "item_index.csv"}')


def main():
    # Inputs: Command-line arguments and a local CSV file with ratings in long form.
    # Outputs: Matrix file and ID mapping files in the selected output directory.
    # Assumptions: Python dependencies are installed; no input files are modified.
    """Run the generic data-preparation workflow for an arbitrary ratings CSV."""
    args = parse_args()
    raw = pd.read_csv(args.input, dtype={args.user_column: str, args.item_column: str})
    df = prepare_ratings(raw, args.user_column, args.item_column,
                         args.rating_column, args.duplicate,
                         args.min_user_ratings, args.min_item_ratings)
    write_outputs(df, args.output_dir, args.format)


if __name__ == '__main__':
    main()
