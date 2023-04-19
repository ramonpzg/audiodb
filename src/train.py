from datasets import load_dataset, Audio, load_from_disk, Dataset, ClassLabel
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import torch
import numpy as np
from transformers import pipeline



def get_data(path_kind, data_dir, split_kind):
    return load_dataset(
        path=path_kind, data_dir=data_dir, split=split_kind
    )

def get_labels(data, col):
    labels = data.features[col].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    num_labels = len(id2label)
    
    return label2id, id2label, num_labels


def split_data(data, ratio=0.2):
    return data.train_test_split(test_size=ratio)


def define_feat_extractor(model):
    return AutoFeatureExtractor.from_pretrained(model)

def features_func(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    return feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt",
        max_length=16000, truncation=True, padding=True
    )

def get_features(data, func, batch_size=50):
    return data.map(func, batched=True, batch_size=batch_size)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


def get_model(model, num_labels, label2id, id2label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AutoModelForAudioClassification.from_pretrained(
        model, num_labels=num_labels, label2id=label2id, id2label=id2label
    ).to(device)


def get_trainer(model, data, targs):
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    return trainer


def save_model(trainer, dir_path):
    trainer.save_model(dir_path)
    print(f"Successfully saved trainer at: {dir_path}")

    
def save_data(data, dir_path):
    data.save_to_disk(dir_path)
    print(f"Successfully saved data at: {dir_path}")


if __name__ == "__main__":
    
    data = get_data(path_kind="audiofolder", data_dir="data/Audios/", split_kind="train")
    
    label2id, id2label, num_labels = get_labels(data, "label")
    
    data = split_data(data)
    feature_extractor = define_feat_extractor("facebook/wav2vec2-base")
    encoded_data = get_features(data, features_func)
    
    accuracy = evaluate.load("accuracy")
    
    model = get_model("facebook/wav2vec2-base", num_labels, label2id, id2label)
    
    targs = TrainingArguments(
        output_dir="models/mod_checkpoints/", evaluation_strategy="epoch",
        save_strategy="epoch",                learning_rate=3e-5,
        per_device_train_batch_size=32,       gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,        num_train_epochs=10,
        warmup_ratio=0.1,                     logging_steps=10,
        load_best_model_at_end=True,          metric_for_best_model="accuracy",
    )
    
    trainer = get_trainer(model, encoded_data, targs)
    save_model(model, "models/my_model")
    
    save_data(encoded_data, "data/processed/")