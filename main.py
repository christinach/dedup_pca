from json_embedding_parser import JSONEmbeddingParser


def main():
    parser = JSONEmbeddingParser()
    data = parser.parse_and_embed("incremental_sample.json")
    parser.save_embedded_json(data, "incremental_sample_embedded.json")
    embeddings_matrix = parser.get_embeddings_matrix(data)
    print("Embeddings matrix shape:", embeddings_matrix.shape)
    print(embeddings_matrix)
    print("Text Embeddings added and saved.")

if __name__ == "__main__":
    main()
