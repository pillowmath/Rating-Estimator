# Rating-Estimator
This repository provides an implementation of the [rating estimator](https://arxiv.org/abs/2410.00865), a new method of averaging ratings that accounts for raters' differing personal rating scales. The method is inspired by the notion of Wasserstein barycenters in optimal transport. See [here](https://pillowmath.github.io/Ratings/the-problem-with-averaging-ratings.html) for easy-to-understand explanations of why we need a new method of averaging ratings and how the rating estimator works.

## Usage
The repository contains python code for calculating the rating estimator and comparing it to average ratings for the following dataset (https://www.kaggle.com/datasets/azathoth42/myanimelist) or on your own rating data.

You can download the dataset and run the calculations yourself, or you can read in your own dataset. The main file for calculating the rating estimator is rating_estimator.py; you just need a matrix of ratings (with raters as rows and items as columns)!

For very large datasets, where the rating matrix may not fit into memory, use rating_estimator_sparse.py, which uses SciPy sparse matrix formats to compute the rating estimator; for this file, you just need a matrix of ratings as an .npz file and a .csv file listing of your item ids (in the same order as the columns of your matrix).

## Files
* The files in the "Build Rating Matrix" folder take in various datasets and convert the raw data into a matrix (or sparse matrix) of ratings, where the rows are raters and the columns are items.

* rating_estimator.py takes in a matrix of ratings (with raters as rows and items as columns) as a .csv file and outputs .csv files for average ratings, primitive ratings, and rating estimator ratings.

* rating_estimator_sparse.py takes in a sparse matrix of ratings (with raters as rows and items as columns) as a .npz file and a list of item ids as a .csv file and outputs .csv files for average ratings, primitive ratings, and rating estimator ratings.

* plot_ratings.py plots the average ratings, primitive ratings, and rating estimator ratings from their .csv files.

* kendall_w.py calculates two statistics, analogues of Kendall's W, which represent the degree of inter-rater agreement in personal rating scales and in overall rating profiles. These are numbers between 0 and 1 (0 meaning no agreement, 1 meaning perfect agreement).

* kendall_w_sparse.py is the same as kendall_w.py, except it takes in a sparse matrix (.npz) instead of a dense matrix (.csv).

* prediction_comparison.py takes in a rating matrix (.csv) and compares the performance the mean vs the rating estimator on a task of predicting user ratings.

* prediction_comparison_sparse.py takes in a sparse rating matrix (.npz) and compares the performance the mean vs the rating estimator on a task of predicting user ratings.

* scale_and_reverse_simulation.py applies the average and the rating estimator to simulated rating data from a "scale and reverse" model and outputs a plot comparing the two results.

* quality_plus_noise_simulation.py applies the average and the rating estimator to simulated rating data from a "quality plus noise" model and outputs a plot comparing the two results.

* quality_plus_noise_simu_structured_missingness.py applies the average and the rating estimator to simulated rating data from a "quality plus noise" model with various choices of structured missingness in the data and outputs a plot comparing the two results.


## Reference
If you found this code helpful, please cite my paper introducing the rating estimator:

Daniel Raban. "How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters" ([Arxiv preprint](https://arxiv.org/abs/2410.00865))

@article{raban2024should,
  title={How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters},
  author={Raban, Daniel},
  journal={arXiv preprint arXiv:2410.00865},
  year={2024}
}
