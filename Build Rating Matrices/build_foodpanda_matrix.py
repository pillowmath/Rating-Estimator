import numpy as np
import pandas as pd
import ast

#First, we read in the dataset.
df = pd.read_csv('BDFoodSent-334k.csv')

# Convert the string in 'ratings' to a list of dictionaries using ast.literal_eval
df['ratings'] = df['ratings'].apply(ast.literal_eval)

# Define a function to extract the restaurant_food score
def get_restaurant_score(rating_list):
    for item in rating_list:
        if item['topic'] == 'restaurant_food':
            return item['score']
    return None  # in case 'restaurant_food' is not found

# Apply the function to create a new column
df['restaurant_score'] = df['ratings'].apply(get_restaurant_score)

df = df.rename(columns = {"name" : "restaurant_id", "reviewerName" : "user_id"})        
        
#We process the data into a (sparse) matrix of ratings with users for rows and books for columns. We remove restaurants which have been rated by < 10 users and then remove users who have rated < 10 restaurants.
rating_matrix = df.pivot_table(index='user_id', columns='restaurant_id', values='restaurant_score', aggfunc='last')
rating_matrix = rating_matrix[rating_matrix.columns[np.logical_xor(rating_matrix.isnull(),1).astype(int).sum() >= 10]]
rating_matrix = rating_matrix[np.logical_xor(rating_matrix.isnull(),1).astype(int).sum(axis=1) >= 10]

#Now we save the ratings matrix to a csv file.
rating_matrix.to_csv('rating_matrix.csv')