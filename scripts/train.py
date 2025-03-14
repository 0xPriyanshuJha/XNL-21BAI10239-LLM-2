from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "models/checkpoints"


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    fp16=torch.cuda.is_available(),
)

# Load processed dataset
train_data = load_dataset("json", data_files="data/processed/train_processed.json")["train"]
val_data = load_dataset("json", data_files="data/processed/val_processed.json")["train"]

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)


trainer.train()


model.save_pretrained("models/trained_model")

print("Model training completed!")
