import numpy as np
import pandas as pd
from itertools import product

#Read in the average ratings, rating estimator ratings, BTL rankings, and counts for pairwise winners
pairwise_counts = pd.read_csv('pairwise_counts.csv',index_col = 0)
average_ratings = pd.read_csv('average_ratings.csv',index_col = 0)
aggregate_ratings = pd.read_csv('target_ratings_revised.csv',index_col = 0)
btl_ratings = pd.read_csv('stat_dist.csv',index_col = 0)

number_of_items = 0
average_count = 0
aggregate_count = 0
btl_count = 0

for ind in pairwise_counts.index:
    a = int(eval(ind)[0])
    b = int(eval(ind)[1])
    if (average_ratings.loc[a,"0"] > average_ratings.loc[b,"0"]) and pairwise_counts.loc[ind,"first_wins"] == 1:
        average_count += 1
    elif (average_ratings.loc[a,"0"] < average_ratings.loc[b,"0"]) and pairwise_counts.loc[ind,"first_wins"] == 0:
        average_count += 1
    
    if (aggregate_ratings.loc[a,"value"] > aggregate_ratings.loc[b,"value"]) and pairwise_counts.loc[ind,"first_wins"] == 1:
        aggregate_count += 1
    elif (aggregate_ratings.loc[a,"value"] < aggregate_ratings.loc[b,"value"]) and pairwise_counts.loc[ind,"first_wins"] == 0:
        aggregate_count += 1
        
    if (btl_ratings.loc[a,"0"] > btl_ratings.loc[b,"0"]) and pairwise_counts.loc[ind,"first_wins"] == 1:
        btl_count += 1
    elif (btl_ratings.loc[a,"0"] < btl_ratings.loc[b,"0"]) and pairwise_counts.loc[ind,"first_wins"] == 0:
        btl_count += 1
        
    if pairwise_counts.loc[ind,"first_wins"] == 1 or pairwise_counts.loc[ind,"first_wins"] == 0:
        number_of_items += 1

#Output how often each ranking aligns with the majority winner in pairwise comparisons
print("average = %f" %(average_count / number_of_items))
print("aggregate = %f" %(aggregate_count / number_of_items))
print("btl = %f" %(btl_count / number_of_items))


