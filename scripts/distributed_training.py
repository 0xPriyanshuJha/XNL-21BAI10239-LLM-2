import torch
import deepspeed
import argparse
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset

def get_training_args(output_dir, batch_size, num_epochs):
    """ Returns training arguments for DeepSpeed & FSDP """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        report_to="wandb", 
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        deepspeed="ds_config.json"  
    )

def train(model_name, dataset_path, output_dir, batch_size=8, num_epochs=3):
    """ Trains the model using DeepSpeed & FSDP """
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    train_data = load_dataset("json", data_files=dataset_path)["train"]

    training_args = get_training_args(output_dir, batch_size, num_epochs)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data
    )


    trainer.train()


    model.save_pretrained(f"{output_dir}/trained_model")
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn", help="Hugging Face model name")
    parser.add_argument("--dataset_path", type=str, default="data/processed/train_processed.json", help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="models/distributed", help="Output directory for model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()
    train(args.model_name, args.dataset_path, args.output_dir, args.batch_size, args.num_epochs)
