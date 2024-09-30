import pandas as pd
import numpy as np
import time

#Read in your rating matrix and the lists of average ratings and rating estimator ratings
avg_rat = pd.read_csv("average_ratings.csv", index_col = 0)
agg_rat = pd.read_csv("rating_estimator_ratings.csv", index_col = 0)
rating_matrix = pd.read_csv("rating_matrix.csv", index_col = 0)

avg_rat = avg_rat.sort_values(by = ["0"], ascending = False)
agg_rat = agg_rat.sort_values(by = ["value"], ascending = False)

#Set the number of items to be considered (i.e. the top n items in each ranking)
number_of_items_considered = 100

#Extract top n anime from each ranking and combine them into a list
top_n_avg = list(pd.DataFrame(avg_rat.index).iloc[0:number_of_items_considered,0])
top_n_avg = [str(item) for item in top_n_avg]
top_n_agg = list(pd.DataFrame(agg_rat.index).iloc[0:number_of_items_considered,0])
top_n_agg = [str(item) for item in top_n_agg]
total_top = list(set(top_n_avg).union(top_n_agg))
total_top = [str(item) for item in total_top]

relevant_matrix = rating_matrix[total_top].dropna(how = "all")

print("\n")



#Utility from averaging ratings without any processing
print("Average-rating average-utility: " + str(relevant_matrix[top_n_avg].mean(axis=1).mean(axis=0)))
print("Aggregate-rating average-utility: " + str(relevant_matrix[top_n_agg].mean(axis=1).mean(axis=0)))




#Utility from averaging quantiles
copy_relevant_matrix = pd.merge(rating_matrix[top_n_avg],rating_matrix[top_n_agg], left_index = True, right_index = True, suffixes = ('', '_dup'))


def replace_by_quantile(row):
    sorted_row = np.sort(row.dropna())
    
    def quantile_fn(list,value):
        if np.isnan(value):
            return None
        else:
            return np.searchsorted(list,value, side = "right")/len(list)
    
    return row.apply(lambda x : quantile_fn(sorted_row,x))
    

copy_relevant_matrix = copy_relevant_matrix.apply(replace_by_quantile,axis=1).dropna(how = "all")


print("Average-rating quantile-utility: " + str(copy_relevant_matrix[top_n_avg].mean(axis = 1).mean(axis=0)))
print("Aggregate-rating quantile-utility: " + str(copy_relevant_matrix[top_n_agg].mean(axis = 1).mean(axis=0)))




#Binary utility for geq mean
def change_values(x):
    if pd.isna(x):
        return x
    elif x >= 0:
        return 1
    else:
        return 0

avg_row_means = relevant_matrix[top_n_avg].mean(axis=1)
avg_geq_mean = relevant_matrix[top_n_avg].sub(avg_row_means,axis=0).applymap(change_values)
agg_row_means = relevant_matrix[top_n_agg].mean(axis=1)
agg_geq_mean = relevant_matrix[top_n_agg].sub(agg_row_means,axis=0).applymap(change_values)

print("Average-rating binary-utility: " + str(avg_geq_mean.mean(axis=1).mean(axis=0)))
print("Aggregate-rating binary-utility: " + str(agg_geq_mean.mean(axis=1).mean(axis=0)))

