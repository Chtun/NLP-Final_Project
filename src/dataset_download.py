import kagglehub
import os
import shutil
from config import DATA_FOLDER

# Download latest version
path = kagglehub.dataset_download("doctri/microsoft-research-paraphrase-corpus")

print("Path to dataset files:", path)

# If the files are in a directory, move them to the specified folder
if os.path.isdir(path):
    shutil.move(path, DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")