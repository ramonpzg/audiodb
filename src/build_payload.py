import torch
from train import get_data
import pandas as pd
from faker import Faker
from random import randint

def get_payload(data, cols):
    pass

def save_payloads(data, cols, file_path_name):
    data[cols].to_json(file_path_name)

def add_rename_cols(data, paths, mapping):
    data = data.rename_column("label", "genre")
    data = data.to_pandas()
    data['audio_path'] = paths
    data["idx"] = [randint(10_000, 99_999) for _ in range(len(data))]
    data["artist"] = [fake.name() for _ in range(len(data))]
    data['genre'] = data['genre'].map(mapping)
    return data

if __name__ == "__main__":
    
    data = get_data(path_kind="audiofolder", data_dir="data/processed/", split_kind="train")
    
    fake = Faker()
    mapping = {0: 'Bachata', 1: 'Cumbia', 2: 'Merengue', 3: 'Salsa', 4: 'Vallenato'}
    
    data = add_rename_cols(data, paths, mapping)
    
    payloads = save_payloads(
        data, 
        ["idx", 'genre', "artist", 'audio_path'],
        "data/payload/payload.json"
    )