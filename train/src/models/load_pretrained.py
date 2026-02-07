import yaml
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_mt5_base_model(config_path=None, force_device=None):
    """
    Downloads and loads the T5 model and tokenizer based on a config file.
    Moves the model to GPU if available.
    """
    
    # Determine the absolute path to the config file if not provided
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
        config_path = os.path.join(project_root, "train", "configs", "model_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = load_config(config_path)
    model_name = config['model']['name']
    save_dir = config['model']['save_dir']
    
    if not os.path.isabs(save_dir):
         current_dir = os.path.dirname(os.path.abspath(__file__))
         project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
         save_dir = os.path.join(project_root, save_dir)

    # Diagnostics
    print(f"--- PyTorch GPU Diagnostics ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Current device index: {torch.cuda.current_device()}")
    print(f"-------------------------------")

    if force_device:
        device = torch.device(force_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Downloading model '{model_name}' to '{save_dir}'...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
    else:
        print(f"Loading model from local directory: {save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)
    
    model.to(device)
    print(f"Model successfully moved to: {device}")
        
    return tokenizer, model, device

if __name__ == "__main__":
    try:
        tokenizer, model, device = load_mt5_base_model()
        print(f"Model and tokenizer ready on {device}.")
    except Exception as e:
        print(f"An error occurred: {e}")