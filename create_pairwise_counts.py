import numpy as np
import pandas as pd

#Read in your rating matrix
rating_matrix = pd.read_csv('rating_matrix.csv')

column_pairs = [(col1, col2) for col1, col2 in product(rating_matrix.columns.drop(['username']), repeat=2) if col1 != col2]

pairwise_counts = pd.DataFrame(index=column_pairs)

pairwise_counts["former_count"] = [((rating_matrix[col1].notnull()) & (rating_matrix[col2].notnull()) & (rating_matrix[col1] > rating_matrix[col2])).sum() for col1, col2 in pairwise_counts.index]

pairwise_counts["latter_count"] = [((rating_matrix[col1].notnull()) & (rating_matrix[col2].notnull()) & (rating_matrix[col1] < rating_matrix[col2])).sum() for col1, col2 in pairwise_counts.index]

# Set first_wins to -1 when former_count equals latter_count and both are not zero
pairwise_counts.loc[(pairwise_counts["former_count"] == pairwise_counts["latter_count"]) & (pairwise_counts["former_count"] != 0), "first_wins"] = -1

# Set first_wins to -2 when former_count and latter_count are both zero
pairwise_counts.loc[(pairwise_counts["former_count"] == 0) & (pairwise_counts["latter_count"] == 0), "first_wins"] = -2

# For the rest, set first_wins = 0 if former_count > latter_count and = 0 former_count < latter_count
pairwise_counts["first_wins"].fillna((pairwise_counts["former_count"] > pairwise_counts["latter_count"]).astype(int), inplace=True)

pairwise_counts.to_csv('pairwise_counts.csv')