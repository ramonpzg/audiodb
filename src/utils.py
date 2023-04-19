from train import get_data
import pandas as pd

def get_paths(data):
    if data["test"][0]:
        data = concatenate_datasets([data['train'], data['test']])
    return (
        data.select_columns("audio")
            .to_pandas()['audio']
            .apply(lambda x: x['path'])
    )


def save_paths(df, file_path_name):
    df.to_parquet(file_path_name)
    print(f"Successfully saved paths at: {file_path_name}")

if __name__ == "__main__":
    
    data = get_data(path_kind="audiofolder", data_dir="data/processed/", split_kind="train")
    save_paths(df, "data/external/paths.parquet")