import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer

DEFAULT_MODEL = "EleutherAI/gpt-neo-2.7B"  # Default to GPT-Neo 2.7B

def load_model_config(config_path):
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        config: Dictionary with configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(modelm=DEFAULT_MODEL, model_type="causal", device="cuda"):
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    try:
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                modelm,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                modelm,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = model.to(device)
        print(f"Successfully loaded model: {modelm}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a smaller model (DistilGPT-2 or T5-Small)...")
        
        fallback_model = "distilgpt2" if model_type == "causal" else "t5-small"
        model = AutoModelForCausalLM.from_pretrained(fallback_model) if model_type == "causal" else AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
        model = model.to(device)
        return model

def load_tokenizer(modelm=DEFAULT_MODEL):
    try:
        tokenizer = AutoTokenizer.from_pretrained(modelm)
        print(f"Successfully loaded tokenizer: {modelm}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return AutoTokenizer.from_pretrained("gpt2")

def create_model_config(modelm=DEFAULT_MODEL, config_path="config/model_config.yaml"):
    try:
        config = AutoConfig.from_pretrained(modelm)

        config_dict = {
            "model_name": modelm,
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "activation_function": getattr(config, "hidden_act", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Model configuration saved to {config_path}")
    
    except Exception as e:
        print(f"Error creating model configuration: {e}")
