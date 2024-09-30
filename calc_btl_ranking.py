import numpy as np
import pandas as pd
from scipy.linalg import eig 

#Read in the matrix of counts of pairwise winners
pairwise_counts = pd.read_csv('pairwise_counts.csv')
pairwise_counts.rename(columns = {"Unnamed: 0" : "pair_index"},inplace = True)

#Convert the matrix of counts into the transition matrix for the BTL Markov chain
pairwise_counts["first_id"] = pairwise_counts["pair_index"].apply(lambda x : eval(x)[0])
pairwise_counts["second_id"] = pairwise_counts["pair_index"].apply(lambda x : eval(x)[1])
pairwise_counts["entry"] = pairwise_counts["latter_count"].astype(float) / (pairwise_counts["former_count"] + pairwise_counts["latter_count"])

diagonal_rows = pd.DataFrame(pairwise_counts["first_id"].drop_duplicates(), columns = ["first_id"])
diagonal_rows["second_id"] = diagonal_rows["first_id"]
diagonal_rows["entry"] = 0

pairwise_counts["entry"] = pairwise_counts["entry"]

pairwise_counts = pd.concat([diagonal_rows, pairwise_counts])

pairwise_matrix = pairwise_counts.pivot_table(index = "first_id",columns = "second_id", values = "entry")
pairwise_matrix.fillna(value=0, inplace=True)

pairwise_matrix_copy = pairwise_matrix.to_numpy()
one_minus_row_sum = len(diagonal_rows.index) - pairwise_matrix.sum(axis=1)
np.fill_diagonal(pairwise_matrix_copy,one_minus_row_sum.to_numpy())

pairwise_matrix = pd.DataFrame(pairwise_matrix_copy,index = diagonal_rows["first_id"], columns = diagonal_rows["second_id"])
transition_matrix = pairwise_matrix.to_numpy()


#Calculate the stationary distribution of the Markov chain by finding an eigenvector for eigenvalue 1.
S, U = eig(transition_matrix.T)
print(S)
print(U)
stationary = np.array(U[:,np.where(np.abs(S-len(pairwise_matrix.index)) < 1e-8)[0][0]].flat)
stationary = stationary / np.sum(stationary)

labeled_stationary = pd.Series(data = stationary, index = pairwise_matrix.index)

#Output unsorted and sorted BTL rankings
labeled_stationary.to_csv('stat_dist.csv')

sorted_stationary = labeled_stationary.sort_values(ascending = False)

sorted_stationary.to_csv('sorted_stat_dist.csv')