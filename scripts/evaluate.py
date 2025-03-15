import os
import torch
import json
import yaml
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from scripts.preprocessing import load_dataset 

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate(model, tokenizer, dataloader, device):
    """Evaluate model on test dataset"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    return {"loss": avg_loss, "perplexity": perplexity.item()}

if __name__ == "__main__":
    config = load_config("config/config.yaml")

    model_name = config["model_name"]
    model_path = config.get("model_checkpoint", "checkpoints/epoch_latest")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset(config["test_data_path"], tokenizer, config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=SequentialSampler(dataset))

    metrics = evaluate(model, tokenizer, dataloader, device)

    print(f"Evaluation Results: {metrics}")

    with open("results/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
