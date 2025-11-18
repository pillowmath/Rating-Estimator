import numpy as np
import pandas as pd
from scipy.sparse import load_npz

#Read in your rating matrix
R = load_npz('rating_matrix.npz').tocsr().astype(np.float64)

#Read in csv files for primitive and rating estimator ratings
aggregate_ratings = pd.read_csv('rating_estimator_ratings.csv', index_col=0)
primitive_ratings = pd.read_csv('primitive_ratings.csv', index_col=0)

agg_variance  = aggregate_ratings['value'].var(ddof=1)
prim_variance = primitive_ratings['value'].var(ddof=1)

#Calculate the variances of the distributions
n = np.diff(R.indptr)
row_sum = np.asarray(R.sum(axis=1)).ravel()
R2 = R.copy(); R2.data **= 2
row_sumsq = np.asarray(R2.sum(axis=1)).ravel()
row_var = np.full(R.shape[0], np.nan)
mask = n > 1
row_var[mask] = (row_sumsq[mask] - row_sum[mask]**2 / n[mask]) / (n[mask] - 1)
avg_of_personal_variances = np.nanmean(row_var)

#Divide the variances to get the two Kendall W statistics
W_scale  = agg_variance  / avg_of_personal_variances
W_rating = prim_variance / avg_of_personal_variances

#Output statistics 
print("W_scale:", W_scale)
print("W_rating:", W_rating)
