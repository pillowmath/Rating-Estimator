# Rating-Estimator
This repository provides an implementation of the rating estimator, a new method of producing aggregate ratings from personal ratings that accounts for raters' differing personal rating scales. The method is inspired by the notion of Wasserstein barycenters in optimal transport.

## Usage
The repository contains python code for comparing the rating estimator and the average on the following dataset: https://www.kaggle.com/datasets/azathoth42/myanimelist. You can download the dataset and run the calculations yourself, or you can read in your own dataset. The main file for calculating the rating estimator given a matrix of ratings is rating_estimator.py.

## Files
* create_rating_matrix.py takes the linked dataset and outputs a .csv file containing a matrix of ratings.

* rating_estimator.py takes in a matrix of ratings as a .csv file and outputs .csv files for average ratings, primitive ratings, and rating estimator ratings.

* plot_ratings.py contains code to plot the average ratings, primitive ratings, and rating estimator ratings from their .csv files

## Reference
If you found this code helpful, please cite my paper introducing the rating estimator:

Daniel Raban. "How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters"
