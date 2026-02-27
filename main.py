from json_embedding_parser import JSONEmbeddingParser
import numpy as np


def main():
    parser = JSONEmbeddingParser()
    data = parser.parse_and_embed("incremental_sample.json")
    parser.save_embedded_json(data, "incremental_sample_embedded.json")
    embeddings_matrix = parser.get_embeddings_matrix(data)
    print("Embeddings matrix shape:", embeddings_matrix.shape)
    print(embeddings_matrix)
    print("Text Embeddings added and saved.")

    similarities = parser.model.similarity(embeddings_matrix, embeddings_matrix)
    print(similarities)

    np.savetxt("similarities_matrix.csv", similarities, delimiter=",")

    duplicates = parser.find_duplicates(similarities, threshold=0.95)
    print("Duplicate pairs (index):", duplicates)

    print(f"Number of duplicate pairs found: {len(duplicates)}")

    for i, j in duplicates:
        print(f"Document {i}:", data[i])
        print(f"Document {j}:", data[j])
        print("-" * 40)

        id_i = data[i].get("id", f"index_{i}")
        id_j = data[j].get("id", f"index_{j}")
        print(f"Duplicate pair: {id_i} and {id_j}")


if __name__ == "__main__":
    main()
