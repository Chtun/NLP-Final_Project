import kagglehub
import os
import shutil
from defensive_tagging_LLM.config import MSRPC_DATA_FOLDER

# Download latest version
path = kagglehub.dataset_download("doctri/microsoft-research-paraphrase-corpus")

print("Path to dataset files:", path)

# Move the data to the specified data folder.
if os.path.isdir(path):
    shutil.move(path, MSRPC_DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")

