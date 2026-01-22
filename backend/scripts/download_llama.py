import os
import sys
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# 1. Load .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("LLAMA_MODEL_ID")
MODEL_DIR = os.getenv("LLAMA_MODEL_DIR")

# 2. Validate env
if not HF_TOKEN:
    print("Error - HF_TOKEN not found in .env")
    sys.exit(1)

if not MODEL_ID or not MODEL_DIR:
    print("LLAMA_MODEL_ID or LLAMA_MODEL_DIR missing in .env")
    sys.exit(1)

# 3. Check if model already exists
if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
    print(f"Model already exists at {MODEL_DIR}, skip download.")
    sys.exit(0)

# 4. Download model
print(f"Downloading {MODEL_ID} ...")
print(f"Target directory: {MODEL_DIR}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
    resume_download=True
)

print("Download completed successfully.")
