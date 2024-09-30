import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#First, we read in the dataset and remove entries where the user has not rated the anime.
df = pd.read_csv('UserAnimeList.csv')
df = df[df["my_score"] != 0]
        
        
#We process the data into three key data structures: The first is a (sparse) matrix of ratings with users for rows and anime for columns. We remove anime which have been rated by < 10 users and then remove users who have rated < 10 anime.
rating_matrix = df.pivot_table(index='username', columns='anime_id', values='my_score', aggfunc='mean')
rating_matrix = rating_matrix[rating_matrix.columns[np.logical_xor(rating_matrix.isnull(),1).astype(int).sum() >= 10]]
rating_matrix = rating_matrix[np.logical_xor(rating_matrix.isnull(),1).astype(int).sum(axis=1) >= 10]

#Now we save the ratings matrix to a csv file.
rating_matrix.to_csv('rating_matrix.csv')