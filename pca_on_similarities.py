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
from sklearn.metrics.pairwise import euclidean_distances


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# scaler = StandardScaler()
# parser = JSONEmbeddingParser()
# data = parser.parse_and_embed("data_with_embeddings/scsb_nypl_embeddings_1.json")
# embeddings_matrix = parser.get_embeddings_matrix(data)
# data_rescaled = scaler.fit_transform(embeddings_matrix)  # original data matrix

# pca = PCA().fit(data_rescaled)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
# print(f"Number of components for 95% variance: {n_components_95}") # number of components for 95% variance: 221


print(
    f"{timestamp()} Loading similarities matrix from 'similarities_matrix/similarities_incremental_03032026_matrix.csv'..."
)
similarities_df = pd.read_csv(
    "similarities_matrix/similarities_incremental_03032026_matrix.csv", index_col=0
)
print(f"{timestamp()} Similarities matrix loaded.")
print(f"Shape of similarities matrix: {similarities_df.shape}")
X = similarities_df.values
print(X[:5, :5])  # print first 5 rows and columns of the similarities matrix

n_components = 44


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
    plt.title(title + " on similarities_incremental_03032026_matrix.csv")
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


# Plot cumulative explained variance for all components
# The elbow point of the plot indicates the the number of components to retain for a good balance between dimensionality reduction and information retention
plt.figure(figsize=(10, 6))
pca_full = PCA().fit(X)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Number of PCA Components")
plt.grid(True)
plt.tight_layout()
# Calculate and print the number of components needed for 95% explained variance

distances = euclidean_distances(
    X_pca
)  # what else can I use to calculate the distance? cosine similarity?
print("Sample pairwise distances (first 5):", distances[0, 1:6])
print("Max distance:", np.max(distances))
print("Min distance (excluding zero):", np.min(distances[distances > 0]))

# Find duplicates. Change the threshold until we find a reasonable amount of duplicates
# X_pca: shape (n_samples, n_components)
distances = euclidean_distances(
    X_pca
)  # what else can I use to calculate the distance? cosine similarity?
duplicate_pairs = []
threshold = 0.74  # Adjust this value
for i in range(distances.shape[0]):
    for j in range(i + 1, distances.shape[1]):
        if distances[i, j] < threshold:
            duplicate_pairs.append((i, j))
print(f"Found {len(duplicate_pairs)} potential duplicate pairs in PCA space.")
if duplicate_pairs:
    print("Duplicate pair indices and record IDs:")
    # Load original data to get record IDs
    import json

    with open("fixed_json/incremental_fixed_03032026.json", "r") as f:
        original_data = json.load(f)
    for i, j in duplicate_pairs:
        id_i = original_data[i].get("id", f"index_{i}")
        id_j = original_data[j].get("id", f"index_{j}")
        print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")

threshold = 0.95
n_components_95 = np.argmax(cumulative_variance >= threshold) + 1
print(
    f"Number of components needed for {int(threshold * 100)}% explained variance: {n_components_95}"
)  # number of components: 44
plt.show()

print("Showing plots...")
plt.show()
