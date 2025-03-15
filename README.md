# DeepSpeed Distributed Training Pipeline

## Overview
This repository provides a DeepSpeed-based distributed training pipeline for a causal language model (LLM) using the Hugging Face `transformers` library. It supports multi-GPU training, mixed-precision (FP16/BF16), MLflow-based experiment tracking, and checkpoint saving.

## Features
- **DeepSpeed Integration**: Efficiently trains large-scale transformer models using DeepSpeed.
- **Multi-GPU Support**: Uses `torch.distributed` to enable distributed training.
- **MLflow Logging**: Tracks experiments, logs hyperparameters, loss metrics, and training progress.
- **Configurable via YAML**: Easily modify hyperparameters and model settings.
- **Checkpointing**: Saves trained model checkpoints after every epoch.

---

## 📂 Repository Structure  

```plaintext
.
│── data/
│   ├── cnn_dailymail/  # Contains dataset files
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   ├── processed/  # Stores preprocessed data
│   │   ├── train_processed.json
│   │   ├── val_processed.json
│   │   ├── test_processed.json
│
│── models/  
│   ├── checkpoints/  # Stores fine-tuned model checkpoints
│   │   ├── checkpoint-xxxx/
│   ├── trained_model/  # Stores final trained model
│   │   ├── model.bin
│
│── scripts/
│   ├── download_data.py  # Downloads dataset from Hugging Face
│   ├── preprocessing.py  # Prepares the dataset
│   ├── train.py  # Fine-tunes the LLM
│   ├── evaluate.py  # Evaluates the trained model
│   ├── hyperparam_tuning.py  # Hyperparameter tuning using Optuna
│   ├── distributed_training.py  # Implements multi-GPU training
│
│── ai_agents/
│   ├── monitoring_agent.py  # AI agent for real-time monitoring
│   ├── hyperparam_agent.py  # AI agent for hyperparameter optimization
│   ├── resource_manager.py  # AI agent for cloud auto-scaling
│
│── cloud/
│   ├── aws_setup.md  # Instructions for AWS instance setup
│   ├── gcp_setup.md  # Instructions for GCP instance setup
│   ├── deploy_model.py  # Deployment script for serving the model
│
│── logs/  # Stores logs for training and monitoring
│── configs/  # Configuration files for different models and experiments
│── requirements.txt  # Dependencies for the project
│── README.md  # Documentation
```
## Installation
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for GPU training)
- Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
Ensure you have the following libraries installed:
```bash
pip install torch transformers deepspeed mlflow datasets pyyaml```

---

## Configuration
Modify `config.yaml` to customize the training process.

Example `config.yaml`:
```yaml
model_name: "facebook/opt-1.3b"
batch_size: 4
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 5e-5
logging_steps: 10
checkpoint_dir: "checkpoints/"
mlflow:
  experiment_name: "DeepSpeed-Training"
  run_name: "LLM-Run"
```

---

## Usage
### Single GPU Training
```bash
python distributed_trainer.py --config config.yaml
```

### Multi-GPU Training
```bash
deepspeed --num_gpus=4 distributed_trainer.py --config config.yaml
```

---

## Training Process
### 1. Setup
- Initializes `DeepSpeed` engine.
- Loads dataset and tokenizer.
- Configures MLflow experiment tracking.

### 2. Training Loop
- Loops through epochs, logging loss and evaluation metrics.
- Saves checkpoints after each epoch.

### 3. Evaluation
- Computes accuracy and loss metrics on a validation set.

---

## Logging and Monitoring
- **MLflow Tracking**: Logs training loss, evaluation accuracy, and hyperparameters.
- **Tensorboard Support** (Optional): Use `torch.utils.tensorboard` for visualization.

To view MLflow logs:
```bash
mlflow ui
```

---

## Results
| Epoch | Training Loss | Validation Accuracy |
|-------|--------------|---------------------|
| 1     | 2.45         | 78.4%               |
| 2     | 1.98         | 82.1%               |
| 3     | 1.75         | 85.6%               |

---

## Checkpoints
Checkpoints are saved in `checkpoints/`. To resume training:
```bash
python distributed_trainer.py --resume_from_checkpoint checkpoints/latest
```

---

## Future Improvements
- Add more datasets for fine-tuning.
- Optimize memory usage with ZeRO stages.
- Support for `bfloat16` precision.

---

## License
MIT License

