import pandas as pd
from datasets import load_from_disk, concatenate_datasets
from train import get_data

def get_paths(data):
    try:
        data = concatenate_datasets([data['train'], data['test']])
        return (
            data.select_columns("audio")
                .to_pandas()['audio']
                .apply(lambda x: x['path'])
        ).to_frame()
    except:
        return (
            data.select_columns("audio")
                .to_pandas()['audio']
                .apply(lambda x: x['path'])
        ).to_frame()

def save_paths(df, file_path_name):
    df.to_parquet(file_path_name)
    print(f"Successfully saved paths at: {file_path_name}")

if __name__ == "__main__":
    
    data = get_data(path_kind="audiofolder", data_dir="data/Audios/", split_kind="train")
    data = get_paths(data)
    save_paths(data, "data/external/paths.parquet")