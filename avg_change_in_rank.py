import numpy as np
import pandas as pd

#Read in the ratings for the average, the rating estimator, and the BTL Markov chain estimator
average_ratings = pd.read_csv('average_ratings.csv').sort_values(by = ["0"])
aggregate_ratings = pd.read_csv('rating_estimator_ratings.csv').sort_values(by = ["value"])
btl_ratings = pd.read_csv('stat_dist.csv').sort_values(by = ["0"])

average_ratings.rename(columns = {'Unnamed: 0' : 'id' },inplace = True)
aggregate_ratings.rename(columns = {'Unnamed: 0' : 'id' },inplace = True)
btl_ratings.rename(columns = {'first_id' : 'id' },inplace = True)

#Calculate the ranks for each item to obtain a ranking from the ratings
average_ratings["average_rank"] = range(len(average_ratings.index))
aggregate_ratings["aggregate_rank"] = range(len(aggregate_ratings.index))
btl_ratings["btl_rank"] = range(len(btl_ratings.index))

merged_1 = pd.merge(average_ratings,aggregate_ratings,on = "id")
merged_2 = pd.merge(merged_1,btl_ratings,on = "id")

#Calculate the normalized average change in ranking between the different ranked lists
dist_avg_agg = (merged_2["average_rank"] - merged_2["aggregate_rank"]).abs().mean()/(len(average_ratings.index)-1)
dist_avg_btl = (merged_2["average_rank"] - merged_2["btl_rank"]).abs().mean()/(len(average_ratings.index)-1)
dist_agg_btl = (merged_2["aggregate_rank"] - merged_2["btl_rank"]).abs().mean()/(len(average_ratings.index)-1)

print("Dist(avg,rat) = %f" %(dist_avg_agg))
print("Dist(avg,btl) = %f" %(dist_avg_btl))
print("Dist(rat,btl) = %f" %(dist_agg_btl))

