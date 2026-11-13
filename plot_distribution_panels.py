#!/usr/bin/env python3
"""Plot three distribution panels used in the JASA paper's dataset figures.

Run after rating_estimator.py or rating_estimator_sparse.py has created its
average_ratings.csv, primitive_ratings.csv, and rating_estimator_ratings.csv files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    # Inputs: CLI options for results directory, output image, dataset label,
    #         histogram bin count, and optional fixed x-axis bounds.
    # Outputs: argparse.Namespace with parsed plotting arguments.
    # Assumptions: Python and Matplotlib are installed; input CSVs exist.
    """Read command-line options for the three-panel distribution figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('rating_distributions.png'))
    parser.add_argument('--dataset', default='')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--xmin', type=float)
    parser.add_argument('--xmax', type=float)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error('--bins must be positive')
    if args.xmin is not None and args.xmax is not None and args.xmin >= args.xmax:
        parser.error('--xmin must be smaller than --xmax')
    return args


def get_scores(results_dir, filename):
    # Inputs: Directory containing output CSVs, and the desired CSV filename.
    # Outputs: Numeric pandas Series of scores with missing rows discarded.
    # Assumptions: Each CSV contains a row-index column and one numeric score column.
    """Read the single score column from an estimator output CSV file."""
    frame = pd.read_csv(results_dir / filename, index_col=0)
    if frame.shape[1] != 1:
        raise ValueError(f'Expected exactly one score column in {filename}')
    return pd.to_numeric(frame.iloc[:, 0], errors='raise').dropna()


def main():
    # Inputs: Three estimator output CSV files and optional CLI plot settings.
    # Outputs: A three-panel PNG/SVG/PDF file at --output (format by suffix).
    # Assumptions: Files were produced using the same input user-item rating matrix.
    """Render average, primitive, and final-estimator rating distributions."""
    args = parse_args()
    specs = [
        ('average_ratings.csv', 'Average ratings for all items'),
        ('primitive_ratings.csv', 'Primitive ratings for all items'),
        ('rating_estimator_ratings.csv', 'Rating estimator scores for all items'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)
    for ax, (filename, title) in zip(axes, specs):
        values = get_scores(args.results_dir, filename)
        ax.hist(values, bins=args.bins)
        ax.set_title(title, fontsize=9)
        if args.xmin is not None or args.xmax is not None:
            ax.set_xlim(left=args.xmin, right=args.xmax)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.2)
    if args.dataset:
        fig.suptitle(args.dataset, fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
