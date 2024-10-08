# Rating-Estimator
This repository provides an implementation of the [rating estimator](https://arxiv.org/abs/2410.00865), a new method of producing aggregate ratings from personal ratings that accounts for raters' differing personal rating scales. The method is inspired by the notion of Wasserstein barycenters in optimal transport.

## Usage
The repository contains python code for comparing the rating estimator and the average on the following dataset: https://www.kaggle.com/datasets/azathoth42/myanimelist. You can download the dataset and run the calculations yourself, or you can read in your own dataset. The main file for calculating the rating estimator given a matrix of ratings is rating_estimator.py.

## Files
* create_ratings_matrix.py takes the linked dataset and outputs a .csv file containing a matrix of ratings.

* rating_estimator.py takes in a matrix of ratings as a .csv file and outputs .csv files for average ratings, primitive ratings, and rating estimator ratings.

* plot_ratings.py plots the average ratings, primitive ratings, and rating estimator ratings from their .csv files.

* kendall_w.py calculates two statistics, analogues of Kendall's W, which represent the degree of inter-rater agreement in personal rating scales and in overall rating profiles. These are numbers between 0 and 1 (0 meaning no agreement, 1 meaning perfect agreement).

* calc_top_n_utility.py takes in the .csv files for the average and rating estimator ratings and calculates the utility associated to the top n items in each ranking, for various notions of utility.

* create_pairwise_counts.py takes in the matrix of ratings and returns a matrix containing a count of how many times item i beat item j in pairwise comparisons (i.e. when both rated by the same rater).

* calc_btl_ranking.py takes in the matrix of pairwise counts and calculates the BTL ranking for the items using the BTL Markov chain estimator.

* avg_change_in_rank.py takes in the .csv files for the average, rating estimator ratings, and BTL Markov chain rankings and calculates the normalized average change in ranking between the different ranked lists obtained from the ratings.

* pairwise_winner_prediction_percentage.py takes in the .csv files for the average, rating estimator ratings, BTL Markov chain rankings, and a matrix of counts for pairwise winners and calculates how often each ranking aligns with the majority winner in pairwise comparisons.

* simulated_ratings.py applies the average and the rating estimator to simulated rating data and outputs a plot comparing the two results.

## Reference
If you found this code helpful, please cite my paper introducing the rating estimator:

Daniel Raban. "How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters" ([Arxiv preprint](https://arxiv.org/abs/2410.00865))
