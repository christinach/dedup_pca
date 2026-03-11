import tarfile
import pymarc
import glob
import os, tarfile, re, json
from pymarc import parse_xml_to_array
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MARCXMLEmbeddingParser:
    
    marcxml_dir = "data_marcxml"

    def extract_and_parse_marcxml(self, marcxml_dir, batch_size=10000):
        # import pdb; pdb.set_trace()
        print(f"Looking for tar.gz files in {marcxml_dir}...")
        tar_files = sorted(
            glob.glob(os.path.join(marcxml_dir, "incremental_*_new.tar.gz"))
        )
        print(f"Found {len(tar_files)} tar.gz files in {marcxml_dir}")
        for tar_idx, tar_gz_path in enumerate(tar_files):
            print(f"Processing {tar_gz_path}")
            extract_dir = "data_marcxml/extracted"
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print(f"Extracted {tar_gz_path} to {extract_dir}")
            # Rename files to add .xml extension if missing
            for f in os.listdir(extract_dir):
                file_path = os.path.join(extract_dir, f)
                if not f.endswith(".xml"):
                    new_file_path = file_path + ".xml"
                    os.rename(file_path, new_file_path)
            xml_files = [
                os.path.join(extract_dir, f)
                for f in os.listdir(extract_dir)
                if f.endswith(".xml")
            ]
            print(f"Found {len(xml_files)} XML files.")
            records = []
            for xml_file in xml_files:
                marc_records = parse_xml_to_array(xml_file)
                for record in marc_records:
                    leader = str(record.leader) if hasattr(record, "leader") else ""
                    control_00x = []
                    numbers_01x_09x = []
                    main_1xx = []
                    title_20x_24x = []
                    edition_25x_28x = []
                    physical_3xx = []
                    series_4xx = []
                    note_5xx = []
                    subject_6xx = []
                    for field in record.get_fields():
                        tag = field.tag
                        if tag.isdigit():
                            tag_int = int(tag)
                            if 0 <= tag_int <= 9:
                                numbers_01x_09x.append(str(field))
                            elif 100 <= tag_int <= 199:
                                main_1xx.append(str(field))
                            elif 200 <= tag_int <= 249:
                                title_20x_24x.append(str(field))
                            elif 250 <= tag_int <= 289:
                                edition_25x_28x.append(str(field))
                            elif 300 <= tag_int <= 399:
                                physical_3xx.append(str(field))
                            elif 400 <= tag_int <= 499:
                                series_4xx.append(str(field))
                            elif 500 <= tag_int <= 599:
                                note_5xx.append(str(field))
                            elif 600 <= tag_int <= 699:
                                subject_6xx.append(str(field))
                            elif 0 <= tag_int <= 99:
                                control_00x.append(str(field))
                    # Extract 001 field directly
                    record_id = None
                    if record['001']:
                        record_id = record['001'].value()
                    record_dict = {
                        "id": record_id,
                        "leader": leader,
                        "control_00x": " ".join(control_00x),
                        "numbers_01x_09x": " ".join(numbers_01x_09x),
                        "main_1xx": " ".join(main_1xx),
                        "title_20x_24x": " ".join(title_20x_24x),
                        "edition_25x_28x": " ".join(edition_25x_28x),
                        "physical_3xx": " ".join(physical_3xx),
                        "series_4xx": " ".join(series_4xx),
                        "note_5xx": " ".join(note_5xx),
                        "subject_6xx": " ".join(subject_6xx),
                    }
                    records.append(record_dict)
            print(f"Parsed {len(records)} MARCXML records from {tar_gz_path}.")
            batch_size = batch_size if batch_size else 10000
            batches = [
                records[i : i + batch_size] for i in range(0, len(records), batch_size)
            ]
            for batch_idx, batch in enumerate(batches):
                print(
                    f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} records."
                )
                batch_embedded = []
                for record in batch:
                    record_id = record.get("id")
                    if not record_id:
                        continue
                    text = " ".join([
                        record["leader"],
                        record["main_1xx"],
                        record["title_20x_24x"],
                        record["edition_25x_28x"],
                        record["physical_3xx"],
                        record["series_4xx"],
                        record["note_5xx"],
                        record["subject_6xx"],
                    ])
                    embedding = self.model.encode(text)
                    batch_embedded.append({"id": record_id, "text_embedding": embedding.tolist()})
                os.makedirs("data_with_embeddings", exist_ok=True)
                batch_json_path = f"data_with_embeddings/marcxml_embeddings_{tar_idx + 1}_batch_{batch_idx + 1}.json"
                with open(batch_json_path, "w") as f:
                    json.dump(batch_embedded, f, indent=2)
                print(f"Saved batch embedding JSON: {batch_json_path}")

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_all_batch_1_embedding_matrix(self):
        """
        Loads all marcxml_embeddings_*_batch_1.json files from data_with_embeddings/,
        creates a matrix of embeddings (IDs as rows, embedding dims as columns),
        and saves the result as similarities_matrix/similarities_batch_1_marcxml_matrix.csv.
        """
        embedding_files = sorted(glob.glob("data_with_embeddings/marcxml_embeddings_*_batch_1.json"))
        all_ids = []
        all_embeddings = []
        for file in embedding_files:
            with open(file, "r") as f:
                data = json.load(f)
            for item in data:
                all_ids.append(item["id"])
                all_embeddings.append(item["text_embedding"])
        if not all_embeddings:
            print("No embeddings found in batch 1 files.")
            return None
        df = pd.DataFrame(all_embeddings, index=all_ids)
        os.makedirs("similarities_matrix", exist_ok=True)
        output_path = "similarities_matrix/similarities_batch_1_marcxml_matrix.csv"
        # Save only the values, no row index and no column headers
        np.savetxt(output_path, df.values, delimiter=",")
        print(f"Saved embedding matrix (no IDs, no headers): {output_path}")
        return df

if __name__ == "__main__":
    parser = MARCXMLEmbeddingParser()
    marcxml_dir = "data_marcxml"
    # parser.extract_and_parse_marcxml(marcxml_dir)
    parser.create_all_batch_1_embedding_matrix()
