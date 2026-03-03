from json_embedding_parser import JSONEmbeddingParser
import numpy as np


def main():
    parser = JSONEmbeddingParser()
    data = parser.parse_and_embed("fixed_json/scsb_nypl_1_fixed.json")
    parser.save_embedded_json(data, "data_with_embeddings/scsb_nypl_embeddings.json")
    embeddings_matrix = parser.get_embeddings_matrix(data)
    print("Embeddings matrix shape:", embeddings_matrix.shape)
    print(embeddings_matrix)
    print("Text Embeddings added and saved.")

    similarities = parser.model.similarity(embeddings_matrix, embeddings_matrix)
    # print(similarities)

    # Save similarities matrix with named columns and rows
    import pandas as pd
    n = similarities.shape[0]
    col_names = [f"embd_{i+1}" for i in range(n)]
    row_names = [f"doc_{i+1}" for i in range(n)]
    df = pd.DataFrame(similarities, columns=col_names, index=row_names)
    df.to_csv("similarities_matrix/similarities_matrix.csv")

    duplicates = parser.find_duplicates(similarities, threshold=0.95)
    print(f"Number of duplicate pairs found: {len(duplicates)}")

    for i, j in duplicates:
        id_i = data[i].get("id", f"index_{i}")
        id_j = data[j].get("id", f"index_{j}")
        print(f"Duplicate pair: {id_i} and {id_j}")

    # # Create a duplicates matrix (1 for duplicate, 0 otherwise)
    # duplicates_matrix = np.zeros_like(similarities)
    # for i, j in duplicates:
    #     duplicates_matrix[i, j] = 1
    #     duplicates_matrix[j, i] = 1  # symmetric

    # # Save duplicates matrix as CSV
    # df_dup = pd.DataFrame(duplicates_matrix, columns=col_names, index=row_names)
    # df_dup.to_csv("similarities_matrix/duplicates_matrix.csv")
    # print("Duplicates matrix saved to similarities_matrix/duplicates_matrix.csv")


if __name__ == "__main__":
    main()
