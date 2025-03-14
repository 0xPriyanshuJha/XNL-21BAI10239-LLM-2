from datasets import load_dataset
import json

import os


dataset = load_dataset("cnn_dailymail", "3.0.0")


save_dir = "data/cnn_dailymail"
os.makedirs(save_dir, exist_ok=True)


for split in ["train", "validation", "test"]:
    data_list = [dict(sample) for sample in dataset[split]] 
    with open(f"{save_dir}/{split}.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)

print("CNN/DailyMail dataset downloaded and saved!")
