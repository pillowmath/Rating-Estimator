import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Pick your favorite lucky random seed
np.random.seed(777)

num_raters = 100

target_ratings = (np.array(range(1000))/ 2000) + 0.25

random_scalings = np.random.normal(loc = 1,scale = 0.25,size = num_raters)
random_flippings = 2 * np.random.binomial(n = 1,p = 0.75,size = num_raters) - 1
random_coeffs = random_scalings * random_flippings

#Calculate the W2 barycenter of raters' personal rating distributions
def calculate_barycenter(matrix):
    # Preset the number of quantiles to be larger than the number of items
    num_quantiles = 50000
    # Calculate quantiles for each user by finding the value at the corresponding index
    quantile_array = np.vstack([ratings[(np.arange(num_quantiles) * len(ratings) // num_quantiles)] for ratings in matrix_array])
    # Calculate the mean across users to find the barycenter
    return np.mean(quantile_array, axis=0)


#The following method takes a user k and a rating value r and returns F_{hat mu}^{-1}(F_k(r)), where F_k is the CDF of user k's rating distribution and hat mu is the W2-barycenter of all users' personal rating distributions.   
def cdf_to_avg_inverse_cdf(input_id,input_value,barycenter_dist):
    input_cdf_value = np.divide(np.searchsorted(matrix_array.loc[input_id],input_value,side = "right"),len(matrix_array.loc[input_id]))
    return barycenter_dist[int(input_cdf_value*50000)-1]


#This method calculates the primitive rating R0 of an item. Since not every user has rated every item, we need to first restrict to the users who have rated the given item.
def primitive_rating(item_id,barycenter_dist):
    relevant_ratings = pd.DataFrame(user_rating_matrix.loc[:,item_id].dropna())
    relevant_ratings['index'] = relevant_ratings.index
    relevant_ratings["value"] = relevant_ratings.apply(lambda row: cdf_to_avg_inverse_cdf(row["index"],row[item_id],barycenter_dist), axis=1).squeeze()
    return np.mean(relevant_ratings.loc[:,"value"])
    
#Calculate the rating estimator ratings of every item
def aggregate_rating(primitive_rating,primitive_rating_list,barycenter_dist):
    input_cdf_value = np.divide(np.searchsorted(primitive_rating_list,primitive_rating,side = "right"),len(primitive_rating_list))
    return barycenter_dist[int(input_cdf_value*50000)-1]

#Now calculate the L2 loss for the average and the rating estimator as a function of the number of raters 
aggregate_l2_losses = []
average_l2_losses = []

for i in range(num_raters):
    user_rating_matrix = pd.DataFrame([((target_ratings-0.5) * coeff)+0.5 for coeff in random_coeffs]).iloc[0:(i+1)]
    

    matrix_array = user_rating_matrix.apply(lambda x: np.sort(x.dropna().to_numpy()), axis=1)    

    barycenter = calculate_barycenter(user_rating_matrix)

    primitive_rating_list = pd.DataFrame(index = user_rating_matrix.columns)
    primitive_rating_list["value"] = primitive_rating_list.index
    primitive_rating_list["value"] = primitive_rating_list["value"].apply(lambda x: primitive_rating(x,barycenter))

    sorted_primitive_rating_list = np.sort(np.array(primitive_rating_list["value"]))
    aggregate_rating_list = primitive_rating_list["value"].apply(lambda x: aggregate_rating(x,sorted_primitive_rating_list,barycenter))

    agg_l2_loss = np.sqrt(np.mean((np.array(aggregate_rating_list) - target_ratings) ** 2))
    avg_l2_loss = np.sqrt(np.mean((np.array(user_rating_matrix.mean(axis = 0)) - target_ratings) ** 2))
    
    aggregate_l2_losses.append(agg_l2_loss)
    average_l2_losses.append(avg_l2_loss)

#Plot the L2 losses
plt.plot([i+1 for i in range(num_raters)],aggregate_l2_losses)
plt.plot([i+1 for i in range(num_raters)],average_l2_losses)
plt.xlabel("Number of raters")
plt.ylabel("L2 loss")
plt.legend(["Rating estimator","Average"])
plt.show()

