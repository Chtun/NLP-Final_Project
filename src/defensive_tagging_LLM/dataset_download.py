import kagglehub
import os
import shutil
from defensive_tagging_LLM.config import MSRPC_DATA_FOLDER, GIGAWORD_DATA_FOLDER, JFLEG_DATA_FOLDER, SST2_DATA_FOLDER, SMS_SPAM_DATA_FOLDER, GLUE_DATA_FOLDER
from datasets import load_dataset

# Microsoft research paraphrase corpus
path = kagglehub.dataset_download("doctri/microsoft-research-paraphrase-corpus")

if os.path.isdir(path):
    shutil.move(path, MSRPC_DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")

print("Path to dataset files:", MSRPC_DATA_FOLDER)

# Gigaword dataset
path = kagglehub.dataset_download("arngowda/gigaword-corpus")

if os.path.isdir(path):
    shutil.move(path, GIGAWORD_DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")

print("Path to dataset files:", GIGAWORD_DATA_FOLDER)

# JFLEG dataset
path = kagglehub.dataset_download("turiabu/jfleg-dataset")

if os.path.isdir(path):
    shutil.move(path, JFLEG_DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")

print("Path to dataset files:", JFLEG_DATA_FOLDER)

# SST2 dataset
dataset = load_dataset("stanfordnlp/sst2")

# Save splits directly into the target folder
for split in dataset:
    save_path = os.path.join(SST2_DATA_FOLDER, f"{split}.csv")
    dataset[split].to_csv(save_path, index=False)

print("Path to dataset files:", SST2_DATA_FOLDER)

# SMS Spam dataset
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

if os.path.isdir(path):
    shutil.move(path, SMS_SPAM_DATA_FOLDER)
else:
    print(f"Expected a directory but got {path}")

print("Path to dataset files:", SMS_SPAM_DATA_FOLDER)

# GLUE dataset
dataset = load_dataset("nyu-mll/glue", "mnli")

# Save splits directly into the target folder
for split in dataset:
    save_path = os.path.join(GLUE_DATA_FOLDER, f"{split}.csv")
    dataset[split].to_csv(save_path, index=False)

print("Path to dataset files:", GLUE_DATA_FOLDER)