from defensive_tagging_LLM.config import *
from huggingface_hub import snapshot_download
import os

# Define the model repository and local directory
repo_id = "Chtun/Defensive_Tagging_LLM"
save_directory = os.path.join(MODEL_WEIGHTS_FOLDER, "Defensive_Tagging_LLM")

# Download the entire model folder
snapshot_download(repo_id=repo_id, local_dir=save_directory)

print(f"Model folder downloaded to {save_directory}")