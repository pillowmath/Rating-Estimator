import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Simulate complete rating matrix
rng = np.random.default_rng(0)
n_users, n_items = 5000, 500
q_items = rng.normal(0.5, 0.15, n_items)
mu = rng.normal(0.0, 0.2, n_users)
sigma = rng.uniform(0.25, 1.75, n_users)
eps = rng.normal(0, 0.15, (n_users, n_items))
matrix = mu[:, None] + sigma[:, None] * (q_items + eps)



#1. User/item structured missingness
#Some users rate more often, and some items are rated more often
user_activity = rng.uniform(0.5, 1.5, n_users)
item_popularity = rng.uniform(0.5, 1.5, n_items)

observation_prob = 0.05 * user_activity[:, None] * item_popularity[None, :]
mask = rng.random((n_users, n_items)) < observation_prob

matrix_user_item = matrix.copy()
matrix_user_item[~mask] = np.nan
rating_df_user_item_preferences = pd.DataFrame(matrix_user_item)


#2. Rating-dependent missingness
#High ratings are more likely to be observed than low ratings
median_rating = np.median(matrix)
observation_prob = np.where(matrix >= median_rating, 0.08, 0.02)
mask = rng.random((n_users, n_items)) < observation_prob

matrix_rating_dependent = matrix.copy()
matrix_rating_dependent[~mask] = np.nan
rating_df_rating_dependent = pd.DataFrame(matrix_rating_dependent)


#3. Per-user rating-dependent missingness
#High ratings are more likely to be observed than low ratings, determined on a per-user basis
user_medians = np.median(matrix, axis=1)
observation_prob = np.where(
    matrix >= user_medians[:, None],
    0.08,
    0.02
)
mask = rng.random((n_users, n_items)) < observation_prob

matrix_per_user_rating_dependent = matrix.copy()
matrix_per_user_rating_dependent[~mask] = np.nan
rating_df_per_user_rating_dependent = pd.DataFrame(matrix_per_user_rating_dependent)


#4. Block missingness
#Users and items belong to one of two groups, and users are more likely to rate items belonging to their own group
user_group = rng.integers(0, 2, n_users)
item_group = rng.integers(0, 2, n_items)

same_group = user_group[:, None] == item_group[None, :]

preference_strength = 0.05
matrix_block = matrix.copy()
matrix_block += np.where(same_group, preference_strength, -preference_strength)

observation_prob = np.where(same_group, 0.08, 0.02)

mask = rng.random((n_users, n_items)) < observation_prob

matrix_block[~mask] = np.nan
rating_df_block = pd.DataFrame(matrix_block)


#COMMENT OUT HERE TO CHANGE MISSINGNESS TYPE

#For user/item structured missingness
#rating_df = rating_df_user_item_preferences

#For rating-dependent missingness
#rating_df = rating_df_rating_dependent

#For per-user rating-dependent missingness
rating_df = rating_df_per_user_rating_dependent

#For block missingness
#rating_df = rating_df_block



#Calculate the W2 barycenter of raters' personal rating distributions
def calculate_barycenter(matrix,matrix_array):
    #Preset the number of quantiles to be larger than the number of items
    num_quantiles = 2 * len(matrix.columns)
    #Calculate quantiles for each user by finding the value at the corresponding index
    quantile_array = np.vstack([ratings[(np.arange(num_quantiles) * len(ratings) // num_quantiles)] for ratings in matrix_array])
    #Calculate the mean across users to find the barycenter
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
plt.figure(figsize=(8, 6))
plt.plot(n_values, mse_prim_list, label="Primitive rating estimator")
plt.plot(n_values, mse_mean_list, label="Average")
plt.xlabel("Number of raters")
plt.ylabel("L2 loss")
plt.legend()
plt.tight_layout()
#plt.title("User/Item Structured Missingness", pad=12)
#plt.title("Rating-Dependent Missingness", pad=12)
plt.title("Per-User Rating-Dependent Missingness", pad=12)
#plt.title("Block Missingness", pad=12)
plt.savefig("structured_missingness.png", dpi=300, bbox_inches="tight")
plt.show()
