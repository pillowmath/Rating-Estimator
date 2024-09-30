import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


#Choose which lines to comment out to plot the rating estimator ratings, primitive ratings, or average ratings

rating_estimator_distribution = pd.read_csv('rating_estimator_ratings.csv',index_col = 0)
#primitive_rating_distribution = pd.read_csv('primitive_ratings.csv',index_col = 0)
#average_distribution = pd.read_csv('average_ratings.csv',index_col = 0)


rating_estimator_distribution.hist(bins = 100)
#primitive_rating_distribution.hist(bins = 100)
#average_distribution.hist(bins = 100)


plt.xlim(1,10)
plt.ylim(0,400)
plt.title("Rating estimator ratings for all items")
#plt.title("Primitive ratings for all items")
#plt.title("Average ratings for all items")
plt.yticks([])
plt.show()