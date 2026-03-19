import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
from sklearn.decomposition import PCA, IncrementalPCA
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sample_embeddings_matrix import SampleEmbeddingsMatrix

class IPCAOnEmbeddings:
    def __init__(self):
        self._embedding_values_cache = None

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate_number_of_components_with_pca(self, variance_threshold=0.95):
        embedding_values = self._embedding_values()
        data_rescaled = self._data_rescaling_with_pca(embedding_values)
        pca = PCA().fit(data_rescaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_features = data_rescaled.shape[1]
        n_components = min(n_components, n_features)
        print(
            f"Number of components for {variance_threshold*100}% variance: {n_components}"
        )
        return n_components

    def _data_rescaling_with_pca(self, embedding_values):
        scaler = StandardScaler()
        data_rescaled = scaler.fit_transform(embedding_values)
        return data_rescaled

    def _embedding_values(self):
        from sklearn.impute import SimpleImputer
        if self._embedding_values_cache is None:
            create_sample_embeddings_matrix = SampleEmbeddingsMatrix()
            create_sample_embeddings_matrix.create_sample()
            embedding_df = pd.read_csv("embeddings_matrix/sample_10000_matrix.csv")
            imputer = SimpleImputer(strategy="mean")
            embedding_values = imputer.fit_transform(embedding_df.values)
            self._embedding_values_cache = embedding_values
        return self._embedding_values_cache
    
    ## IncrementalPCA (IPCA) on embeddings ##
    def calculate_ipca(self, n_components, batch_size):
        print(f"{self.timestamp()} Starting IncrementalPCA on embeddings...")
        # batch_files = sorted(glob.glob("embeddings_matrix/scsb_update_batch_*_matrix.csv"))
        return IncrementalPCA(n_components=n_components, batch_size=batch_size)
     

    def ipca_fit(self, batch_files, n_components, batch_size):
        scaler = StandardScaler()
        first_batch_data = pd.read_csv(batch_files[0]).values
        n_features = first_batch_data.shape[1]
        n_samples_first_batch = first_batch_data.shape[0]
        n_components = min(n_components, n_features, n_samples_first_batch)
        print(f"n_components: {n_components}, n_features: {n_features}, n_samples_first_batch: {n_samples_first_batch}")
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for batch_file in batch_files:
            print(f"Fitting IPCA on {batch_file}")
            batch_data = pd.read_csv(batch_file).values
            batch_data = scaler.fit_transform(batch_data)
            ipca.partial_fit(batch_data)
        return ipca

    def ipca_transform(self, batch_files, ipca):
        scaler = StandardScaler()
        transformed_batches = []
        for batch_file in batch_files:
            print(f"Transforming {batch_file}")
            batch_data = pd.read_csv(batch_file).values
            batch_data = scaler.fit_transform(batch_data)
            x_ipca_batch = ipca.transform(batch_data)
            transformed_batches.append(x_ipca_batch)
        return transformed_batches

    def ipca_combine_transformed_batches(self, transformed_batches, ipca):
        X_ipca = np.vstack(transformed_batches)
        print("Final IPCA shape:", X_ipca.shape)
        print("IPCA transformed data (first 5 rows):")
        print(X_ipca[:5])
        print("Explained variance ratio (IPCA):")
        print(ipca.explained_variance_ratio_)

        print("Cumulative explained variance (IPCA):")
        print(np.cumsum(ipca.explained_variance_ratio_))
    
    def euclidean_distances_in_ipca_space(self, X_ipca):
        distances_ipca = euclidean_distances(X_ipca)
        print("Sample pairwise distances in IPCA space (first 5):", distances_ipca[0, 1:6])
        print("Max distance in IPCA space:", np.max(distances_ipca))
        print(
            "Min distance in IPCA space (excluding zero):",
            np.min(distances_ipca[distances_ipca > 0]),
        )

    def identify_duplicates(euclidean_distances_in_ipca_space):
        threshold_ipca = 2.39  # Adjust this value

        print("threshold_ipca:", threshold_ipca)
        duplicate_ipca_pairs = []
        for i in range(euclidean_distances_in_ipca_space.shape[0]):
            for j in range(i + 1, euclidean_distances_in_ipca_space.shape[1]):
                if euclidean_distances_in_ipca_space[i, j] < threshold_ipca:
                    duplicate_ipca_pairs.append((i, j))
        print(f"Found {len(duplicate_ipca_pairs)} potential duplicate pairs in IPCA space.")
        # Build combined ID list from all batch JSON files
        combined_ids = []
        batch_json_files = sorted(
            glob.glob("data_with_embeddings/scsb_update_*_batch_1.json")
        )
        for batch_json_file in batch_json_files:
            with open(batch_json_file, "r") as f:
                batch_data = json.load(f)
                combined_ids.extend(
                    [item.get("id", f"index_{idx}") for idx, item in enumerate(batch_data)]
                )
        if duplicate_ipca_pairs:
            print("Duplicate pair indices and record IDs (IPCA):")
            for i, j in duplicate_ipca_pairs:
                id_i = combined_ids[i] if i < len(combined_ids) else f"index_{i}"
                id_j = combined_ids[j] if j < len(combined_ids) else f"index_{j}"
                print(f"Pair: ({i}, {j}) -> IDs: {id_i}, {id_j}")

    ### start PCA ###
    # print(f"{timestamp()} Starting standard PCA fit...")
    # print(f"Shape of X before standard PCA: {X.shape}")
    # print(f"{timestamp()} Running standard PCA with n_components={n_components}...")
    # pca = PCA(n_components=n_components)
    # X_pca = pca.fit_transform(X)
    # print(f"{timestamp()} Standard PCA fit complete.")

    # # Plotting (no target labels, so just scatter all points)
    # for X_transformed, title in [(X_pca, "PCA")]:
    #     print(f"Plotting results for {title}...")
    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(X_transformed[:, 0], X_transformed[:, 1], color="navy", lw=2)
    #     plt.title(title + " on similarities_incremental_03032026_matrix.csv")
    #     plt.axis("equal")

    # # X_pca is the transformed data
    # print("PCA transformed data (first 5 rows):")
    # print(X_pca[:5])

    # # Explained variance ratio for each component
    # print("Explained variance ratio:")
    # print(pca.explained_variance_ratio_)

    # # Cumulative explained variance
    # print("Cumulative explained variance:")
    # print(np.cumsum(pca.explained_variance_ratio_))

    # # Plot cumulative explained variance for all components
    # # The elbow point of the plot indicates the the number of components to retain for a good balance between dimensionality reduction and information retention
    # plt.figure(figsize=(10, 6))
    # pca_full = PCA().fit(X)
    # cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    # plt.xlabel("Number of Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title("Cumulative Explained Variance by Number of PCA Components")
    # plt.grid(True)
    # plt.tight_layout()

        print("Sample pairwise distances in IPCA space (first 5):", euclidean_distances_in_ipca_space[0, 1:6])
        print("Max distance in IPCA space:", np.max(euclidean_distances_in_ipca_space))
        print(
            "Min distance in IPCA space (excluding zero):",
            np.min(euclidean_distances_in_ipca_space[euclidean_distances_in_ipca_space > 0]),
        )

        print("Showing plots...")

        plt.show()
