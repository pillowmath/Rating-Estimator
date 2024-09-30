import numpy as np
import pandas as pd

#Read in your rating matrix
rating_matrix = pd.read_csv('rating_matrix.csv',index_col = 0)

#Read in csv files for primitive and 
aggregate_ratings = pd.read_csv('rating_estimator_ratings.csv',index_col = 0)
primitive_ratings = pd.read_csv('primitive_ratings.csv',index_col = 0)

#Calculate the variances of the distributions
agg_variance = aggregate_ratings['value'].var()
prim_variance = primitive_ratings['value'].var()
avg_of_personal_variances = rating_matrix.var(axis=1).mean()

#Divide the variances to get the two Kendall W statistics
W_scale = np.divide(agg_variance,avg_of_personal_variances)
W_rating = np.divide(prim_variance,avg_of_personal_variances)

#Output statistics 
print("W_scale: " + str(W_scale))
print("W_rating: " + str(W_rating))