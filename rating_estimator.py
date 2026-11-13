import numpy as np
import pandas as pd

#Read in a matrix of ratings. The rows are raters and the columns are items.
rating_matrix = pd.read_csv('rating_matrix.csv',index_col = 0)
rating_matrix = rating_matrix.dropna(axis=1, how='all')

# Drop NaN values and convert the matrix rows to sorted arrays
matrix_array = rating_matrix.apply(lambda x: np.sort(x.dropna().to_numpy()), axis=1)

#Calculate the W2 barycenter of raters' personal rating distributions
# Function: calculate_barycenter
# Inputs: matrix: dense user-by-item DataFrame (the module also uses its sorted rows).
# Outputs: 1-D array of averaged empirical quantiles, length twice the number of items.
# Assumptions: Each included user has at least one observed rating; module-level rating_matrix and matrix_array correspond to matrix.
def calculate_barycenter(matrix):
    # Preset the number of quantiles to be larger than the number of items
    num_quantiles = 2 * len(rating_matrix.columns)
    # Calculate quantiles for each user by finding the value at the corresponding index
    quantile_array = np.vstack([ratings[(np.arange(num_quantiles) * len(ratings) // num_quantiles)] for ratings in matrix_array])
    # Calculate the mean across users to find the barycenter
    return np.mean(quantile_array, axis=0)


#The following method takes a user k and a rating value r and returns F_{hat mu}^{-1}(F_k(r)), where F_k is the CDF of user k's rating distribution and hat mu is the W2-barycenter of all users' personal rating distributions.   
# Function: cdf_to_avg_inverse_cdf
# Inputs: input_id: user row identifier; input_value: observed rating; barycenter_dist: barycenter quantile array.
# Outputs: Scalar rating mapped to the barycenter quantile scale.
# Assumptions: input_id occurs in matrix_array; the rating is observed for that user; the array is calculated for the same matrix.
def cdf_to_avg_inverse_cdf(input_id,input_value,barycenter_dist):
    input_cdf_value = np.divide(np.searchsorted(matrix_array.loc[input_id],input_value,side = "right"),len(matrix_array.loc[input_id]))
    return barycenter_dist[int(input_cdf_value*2 * len(rating_matrix.columns))-1]


#This method calculates the primitive rating R0 of an item. Since not every user has rated every item, we need to first restrict to the users who have rated the given item.
# Function: primitive_rating
# Inputs: item_id: item/column identifier; barycenter_dist: barycenter quantile array.
# Outputs: Scalar mean of barycenter-transformed ratings for the item.
# Assumptions: The item has at least one observed rating and all contributing users have nonempty rating rows.
def primitive_rating(item_id,barycenter_dist):
    relevant_ratings = pd.DataFrame(rating_matrix.loc[:,item_id].dropna())
    relevant_ratings['index'] = relevant_ratings.index
    relevant_ratings["value"] = relevant_ratings.apply(lambda row: cdf_to_avg_inverse_cdf(row["index"],row[item_id],barycenter_dist), axis=1).squeeze()
    return np.mean(relevant_ratings.loc[:,"value"])
    
#Calculate the rating estimator ratings of every item
# Function: aggregate_rating
# Inputs: primitive_rating: one item score; primitive_rating_list: sorted array of all item primitive scores; barycenter_dist: quantile array.
# Outputs: Final scalar score after monotone quantile rescaling.
# Assumptions: primitive_rating_list is nonempty and sorted; barycenter grid is based on the same item set.
def aggregate_rating(primitive_rating,primitive_rating_list,barycenter_dist):
    input_cdf_value = np.divide(np.searchsorted(primitive_rating_list,primitive_rating,side = "right"),len(primitive_rating_list))
    return barycenter_dist[int(input_cdf_value*2 * len(rating_matrix.columns))-1]


#Now apply the above methods to calculate the rating estimator    
barycenter = calculate_barycenter(rating_matrix)

primitive_rating_list = pd.DataFrame(index = rating_matrix.columns)
primitive_rating_list["value"] = primitive_rating_list.index
primitive_rating_list["value"] = primitive_rating_list["value"].apply(lambda x: primitive_rating(x,barycenter))

sorted_primitive_rating_list = np.sort(np.array(primitive_rating_list["value"]))
rating_estimator_rating_list = primitive_rating_list["value"].apply(lambda x: aggregate_rating(x,sorted_primitive_rating_list,barycenter))


#Save the average ratings, primitive ratings, and rating estimator ratings to csv files
rating_matrix.mean().to_csv('average_ratings.csv')

primitive_rating_list.to_csv('primitive_ratings.csv')

rating_estimator_rating_list.to_csv('rating_estimator_ratings.csv')

