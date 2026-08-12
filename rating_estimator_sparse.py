import numpy as np
import pandas as pd
from scipy.sparse import load_npz

#Load sparse matrix and item index.
sparse_matrix = load_npz("rating_matrix.npz")
item_index = pd.read_csv("item_index.csv", index_col=0).squeeze("columns")

#Convert the matrix rows to sorted arrays of ratings for each rater.
row_values = [np.sort(sparse_matrix.data[sparse_matrix.indptr[i]:sparse_matrix.indptr[i + 1]].copy()) for i in range(sparse_matrix.shape[0])]

#This method calculate the W2 barycenter of raters' personal rating distributions.
def calculate_barycenter(matrix):
    #Preset the number of quantiles to be larger than the number of items
    num_quantiles = 2 * sparse_matrix.shape[1]
    #Calculate quantiles for each user by finding the value at the corresponding index and store these in a running average
    running_average = np.arange(num_quantiles) * 0
    for i in range(len(row_values)):
        running_average = (i / (i+1)) * running_average + (1/(i+1))*row_values[i][(np.arange(num_quantiles) * len(row_values[i]) // num_quantiles)]
    return running_average


#The following method takes a user k and a rating value r and returns F_{hat mu}^{-1}(F_k(r)), where F_k is the CDF of user k's rating distribution and hat mu is the W2-barycenter of all users' personal rating distributions.   
def cdf_to_avg_inverse_cdf(input_id,input_value,barycenter_dist):
    row = row_values[input_id]
    input_cdf_value = np.divide(np.searchsorted(row,input_value,side = "right"),len(row))
    index = int(input_cdf_value * 2 * sparse_matrix.shape[1]) - 1
    return barycenter_dist[index]


#This method calculates the primitive rating R0 of all items in one row-wise pass through the sparse matrix.
def calculate_primitive_ratings(barycenter_dist):
    num_items = sparse_matrix.shape[1]

    #Running sums and counts for each item.
    item_sums = np.zeros(num_items, dtype=float)
    item_counts = np.zeros(num_items, dtype=int)

    #CSR matrices are stored by row, so process one user at a time.
    for user_id in range(sparse_matrix.shape[0]):
        start = sparse_matrix.indptr[user_id]
        end = sparse_matrix.indptr[user_id + 1]

        item_ids = sparse_matrix.indices[start:end]
        ratings = sparse_matrix.data[start:end]

        for item_id, rating in zip(item_ids, ratings):
            transformed_rating = cdf_to_avg_inverse_cdf(user_id, rating, barycenter_dist)
            
            item_sums[item_id] += transformed_rating
            item_counts[item_id] += 1

    #Calculate the mean transformed rating for each item.
    return np.divide(item_sums, item_counts, out=np.zeros_like(item_sums, dtype=float), where=item_counts != 0)
    
#This method calculates the rating estimator scores of every item
def aggregate_rating(primitive_rating,primitive_rating_list,barycenter_dist):
    input_cdf_value = np.divide(np.searchsorted(primitive_rating_list,primitive_rating,side = "right"),len(primitive_rating_list))
    return barycenter_dist[int(input_cdf_value*2 * sparse_matrix.shape[1])-1]

#We now apply the above methods to calculate the rating estimator. First, we calculate the W2 barycenter of raters' personal rating distributions.
barycenter = calculate_barycenter(row_values)

#Create a list of primitive ratings (one for each item). Then record a version of it, sorted in increasing order.
primitive_ratings = pd.Series(calculate_primitive_ratings(barycenter), index=item_index.index)
sorted_primitive_values = np.sort(primitive_ratings.values)

#Calculate the rating estimator using aggregate_rating.
rating_estimator_ratings = primitive_ratings.apply(lambda x: aggregate_rating(x, sorted_primitive_values, barycenter))

#Calculate the mean of each column of the matrix to later store as average ratings for each item.
col_counts = (sparse_matrix != 0).sum(axis=0).A1
col_sums = sparse_matrix.sum(axis=0).A1
col_means = np.divide(col_sums, col_counts, out=np.zeros_like(col_sums, dtype=float), where=col_counts != 0)

#Save the average ratings, primitive ratings, and rating estimator ratings to csv files.
pd.Series(col_means,index = item_index.index).to_csv('average_ratings.csv')

primitive_ratings.to_frame("value").to_csv('primitive_ratings.csv')

rating_estimator_ratings.to_frame("value").to_csv('rating_estimator_ratings.csv')

