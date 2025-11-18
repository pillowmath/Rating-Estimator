import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Simulate rating matrix with missing ratings
rng = np.random.default_rng(0)
n_users, n_items = 5000, 500
q_items = rng.normal(0.5, 0.15, n_items)
mu = rng.normal(0.0, 0.2, n_users)
sigma = rng.uniform(0.25, 1.75, n_users)
eps = rng.normal(0, 0.15, (n_users, n_items))
matrix = mu[:, None] + sigma[:, None] * (q_items + eps)
mask = rng.random((n_users, n_items)) < 0.05
matrix[~mask] = np.nan
rating_df = pd.DataFrame(matrix)


#Calculate the W2 barycenter of raters' personal rating distributions
def calculate_barycenter(matrix,matrix_array):
    # Preset the number of quantiles to be larger than the number of items
    num_quantiles = 2 * len(matrix.columns)
    # Calculate quantiles for each user by finding the value at the corresponding index
    quantile_array = np.vstack([ratings[(np.arange(num_quantiles) * len(ratings) // num_quantiles)] for ratings in matrix_array])
    # Calculate the mean across users to find the barycenter
    return np.mean(quantile_array, axis=0)


#The following method takes a user k and a rating value r and returns F_{hat mu}^{-1}(F_k(r)), where F_k is the CDF of user k's rating distribution and hat mu is the W2-barycenter of all users' personal rating distributions.   
def cdf_to_avg_inverse_cdf(input_id,input_value,barycenter_dist,matrix_array,rating_matrix):
    input_cdf_value = np.divide(np.searchsorted(matrix_array.loc[input_id],input_value,side = "right"),len(matrix_array.loc[input_id]))
    return barycenter_dist[int(input_cdf_value*2 * len(rating_matrix.columns))-1]


#This method calculates the primitive rating R0 of an item. Since not every user has rated every item, we need to first restrict to the users who have rated the given item.
def primitive_rating(item_id,barycenter_dist,matrix_array,rating_matrix):
    relevant_ratings = pd.DataFrame(rating_matrix.loc[:,item_id].dropna())
    relevant_ratings['index'] = relevant_ratings.index
    relevant_ratings["value"] = relevant_ratings.apply(lambda row: cdf_to_avg_inverse_cdf(row["index"],row[item_id],barycenter_dist,matrix_array,rating_matrix), axis=1).squeeze()
    return np.mean(relevant_ratings.loc[:,"value"])


#Run mse calculation for various numbers of raters
n_values = list(range(50, 5001, 50))
mse_mean_list, mse_prim_list = [], []
    
for n in n_values:
    rat_mat = rating_df.iloc[:n].dropna(axis=1, how="all")
    mat_ary = rat_mat.apply(lambda x: np.sort(x.dropna().to_numpy()), axis=1)
    
    barycenter = calculate_barycenter(rat_mat,mat_ary)
    
    primitive_rating_list = pd.DataFrame(index = rat_mat.columns)
    primitive_rating_list["value"] = primitive_rating_list.index
    primitive_rating_list["value"] = primitive_rating_list["value"].apply(lambda x: primitive_rating(x,barycenter,mat_ary,rat_mat))
    
    mean_list = rat_mat.mean()
    cols = rat_mat.columns.astype(int).to_numpy()
    q_subset = q_items[cols]
    
    mse_mean = np.mean((q_subset - mean_list) ** 2)
    mse_prim = np.mean((q_subset - primitive_rating_list["value"].to_numpy()) ** 2)
    
    mse_mean_list.append(mse_mean)
    mse_prim_list.append(mse_prim)


#Plot results
plt.figure()
plt.plot(n_values, mse_prim_list, label="Primitive rating estimator")
plt.plot(n_values, mse_mean_list, label="Average")
plt.xlabel("Number of raters")
plt.ylabel("L2 loss")
plt.legend()
plt.tight_layout()
plt.show()
