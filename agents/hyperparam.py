import optuna
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from datasets import load_dataset

MODEL_NAME = "facebook/bart-large-cnn"

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


    train_data = load_dataset("json", data_files="data/processed/train_processed.json")["train"]

    training_args = TrainingArguments(
        output_dir="models/optimized",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=3,
        save_total_limit=2,
        evaluation_strategy="epoch",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()
    return trainer.evaluate()["eval_loss"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best Hyperparameters:", study.best_params)
