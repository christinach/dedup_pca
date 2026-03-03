import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm

# # 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# [3, 384]

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#          [0.6660, 1.0000, 0.1411],
#          [0.1046, 0.1411, 1.0000]])


class JSONEmbeddingParser:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def parse_and_embed(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        for entry in data:
            fields = [
                entry.get("isbn_display", ""),
                entry.get("oclc_s", []),
                entry.get("title_display", ""),
                entry.get("call_number_display", ""),
                entry.get("publication_display", ""),
                entry.get("edition_display", ""),
                # entry.get("lcgft_s", ""),
                # entry.get("rbgenr_s", ""),
                entry.get("uniform_title_s", ""),
                entry.get("series_statement_index", ""),
                entry.get("context_title_index", ""),
            ]
            text = " ".join(
                [
                    field if isinstance(field, str) else " ".join(field)
                    for field in fields
                ]
            )
            embedding = self.model.encode(text)
            entry["text_embedding"] = embedding.tolist()
        return data

    def save_embedded_json(self, data, output_path):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_embeddings_matrix(self, data):
        """
        Returns a numpy array of shape (num_documents, embedding_dim)
        """
        embeddings = [
            entry["text_embedding"] for entry in data if "text_embedding" in entry
        ]
        return np.array(embeddings)

    def find_duplicates(self, similarities, threshold=0.95):
        """
        Returns a list of index pairs (i, j) where similarity > threshold and i != j
        """
        duplicates = []
        for i in range(similarities.shape[0]):
            for j in range(i + 1, similarities.shape[1]):
                if similarities[i, j] > threshold:
                    duplicates.append((i, j))
        return duplicates
