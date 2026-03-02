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
    # print("Duplicate pairs (index):", duplicates)

    print(f"Number of duplicate pairs found: {len(duplicates)}")

    for i, j in duplicates:
        # print(f"Document {i}:", data[i])
        # print(f"Document {j}:", data[j])
        # print("-" * 40)

        id_i = data[i].get("id", f"index_{i}")
        id_j = data[j].get("id", f"index_{j}")
        print(f"Duplicate pair: {id_i} and {id_j}")


if __name__ == "__main__":
    main()
