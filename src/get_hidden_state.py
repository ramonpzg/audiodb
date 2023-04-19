import torch
from train import define_feat_extractor
from transformers import AutoModel
from datasets import concatenate_datasets, load_from_disk
import numpy as np


def load_pretrained(model_ckpt, device):
    return AutoModel.from_pretrained(model_ckpt).to(device)

def prep_format(data):
    data.set_format("torch", columns=["label", "input_values"])
    if data["test"][0]:
        data = concatenate_datasets([data['train'], data['test']])
    return data
    

def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in feature_extractor.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

def save_hs(hidden_state, file_path_name):
    np.save(
        file_path_name, 
        np.array(hidden_state["hidden_state"]), 
        allow_pickle=False
    )
    print(f"Successfully saved Hidden State Layer at: {file_path_name}")

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = load_from_disk("data/processed/")
    data = prep_format(data)
    
    feature_extractor = define_feat_extractor("facebook/wav2vec2-base")
    model = load_pretrained("models/my_model", device)
    hidden_state = data.map(extract_hidden_states, batched=True, batch_size=50)
    
    save_hs(hidden_state, 'data/hidden_state/vectors_full.npy')