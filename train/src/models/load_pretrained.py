import yaml
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_t5_vietnamese_model(config_path=None):
    """
    Downloads and loads the T5 model and tokenizer based on a config file.
    """
    
    # Determine the absolute path to the config file if not provided
    if config_path is None:
        # Assuming the script is in train/src/models/
        # and config is in train/configs/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../")) # Go up to meeting-summarization/
        config_path = os.path.join(project_root, "train", "configs", "model_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = load_config(config_path)
    
    model_name = config['model']['name']
    
    # Handle save_dir relative to project root if it's not absolute
    save_dir = config['model']['save_dir']
    if not os.path.isabs(save_dir):
         # Assuming the script is run from project root or save_dir is relative to project root
         # Let's anchor it to the location of the config file or project root for safety.
         # Here I'll anchor it to the directory containing the 'train' folder (project root)
         # calculated above in 'project_root' context, but let's recalculate for robustness
         current_dir = os.path.dirname(os.path.abspath(__file__))
         project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
         save_dir = os.path.join(project_root, save_dir)

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
        
    return tokenizer, model

if __name__ == "__main__":
    try:
        tokenizer, model = load_t5_vietnamese_model()
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")