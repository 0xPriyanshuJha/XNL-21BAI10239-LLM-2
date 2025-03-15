import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_dataset(data_path, tokenizer_name, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    dataset = load_dataset("json", data_files=data_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    class LLM_Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    return LLM_Dataset(tokenized_datasets["train"])

if __name__ == "__main__":
    import yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    dataset = load_dataset(config["train_data_path"], config["model_name"], config["max_length"])
    print(f"Loaded {len(dataset)} training samples")
