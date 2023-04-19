from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import pandas as pd
import numpy as np


def get_vectors_and_payloads(vectors_path, payload_path):
    vectors = np.load(vectors_path)
    payload = pd.read_json(payload_path, orient="records").set_index("idx")
    return vectors, payload


def create_vdb(client, name):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def add_data_to_vdb(name, vectors, payload):
    client.upsert(
        collection_name=name,
        points=models.Batch(
            ids=payload.index.to_list(),
            payloads=payload.to_dict(orient="records"),
            vectors=vectors.tolist()
        ),
    )


if __name__ == "__main__":
    
    vectors_path = 'data/hidden_state/vectors_full.npy'
    payload_path = "data/payloads/payload.json"
    vdb_name = "music_collection"
    
    
    client = QdrantClient("localhost", port=6333)
    
    vectors, payload = get_vectors_and_payloads(vectors_path, payload_path)
    
    create_vdb(client, vdb_name)
    
    add_data_to_vdb(vdb_name, vectors, payload)