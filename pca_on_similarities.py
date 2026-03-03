# PCA on similarities_matrix.csv embeddings
# SPDX-License-Identifier: BSD-3-Clause
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from json_embedding_parser import JSONEmbeddingParser
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# scaler = StandardScaler()
# parser = JSONEmbeddingParser()
# data = parser.parse_and_embed("data_with_embeddings/scsb_nypl_embeddings_1.json")
# embeddings_matrix = parser.get_embeddings_matrix(data)
# data_rescaled = scaler.fit_transform(embeddings_matrix)  # original data matrix

# pca = PCA().fit(data_rescaled)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
# print(f"Number of components for 95% variance: {n_components_95}") # number of components for 95% variance: 221





print(f"{timestamp()} Loading similarities matrix from 'similarities_matrix/similarities_matrix.csv'...")
similarities_df = pd.read_csv("similarities_matrix/similarities_matrix.csv", index_col=0)
print(f"{timestamp()} Similarities matrix loaded.")
print(f"Shape of similarities matrix: {similarities_df.shape}")
X = similarities_df.values
print(X[:5, :5])  # print first 5 rows and columns of the similarities matrix

n_components = 221


# print(f"{timestamp()} Running IncrementalPCA with n_components={n_components}...")
# print(f"Shape of X before IncrementalPCA: {X.shape}")
# print(f"{timestamp()} Starting IncrementalPCA fit...")
# ipca = IncrementalPCA(n_components=n_components, batch_size=100)
# X_ipca = ipca.fit_transform(X)
# print(f"{timestamp()} IncrementalPCA fit complete.")


print(f"{timestamp()} Starting standard PCA fit...")
print(f"Shape of X before standard PCA: {X.shape}")
print(f"{timestamp()} Running standard PCA with n_components={n_components}...")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
print(f"{timestamp()} Standard PCA fit complete.")

# Plotting (no target labels, so just scatter all points)
for X_transformed, title in [(X_pca, "PCA")]:
    print(f"Plotting results for {title}...")
    plt.figure(figsize=(8, 8))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color="navy", lw=2)
    plt.title(title + " on similarities_matrix.csv")
    plt.axis("equal")

# X_pca is the transformed data
print("PCA transformed data (first 5 rows):")
print(X_pca[:5])

# Explained variance ratio for each component
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

# Cumulative explained variance
print("Cumulative explained variance:")
print(np.cumsum(pca.explained_variance_ratio_))

print("Showing plots...")
plt.show()
