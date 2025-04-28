from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class TaggingDataset(Dataset):
    def __init__(self, texts, tags, tokenizer):
        self.texts = texts  # List of texts
        self.tags = tags    # List of corresponding tags for each text (0 for QUERY, 1 for DATA)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)  # Return the number of texts

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get the text
        tag = self.tags[idx]    # Get the tag corresponding to this text
        
        # Tokenize the text (without padding here to match your example)
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")

        # Assign the same tag to all tokens of this text
        tag_ids = torch.full_like(inputs['input_ids'], tag)  # Create a tensor with the same shape as input_ids and fill it with the tag

        return {
            "input_ids": inputs["input_ids"].squeeze(0),  # Remove the batch dimension (if any)
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "tag_ids": tag_ids.squeeze(0)  # Remove batch dimension here too
        }

# Example texts and tags
texts = ["This is a query.", "And this is data."]
tags = [0, 1]  # 0 = QUERY, 1 = DATA

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("llama-2-7b-chat", local_files_only=True)

# Create the dataset and dataloader
dataset = TaggingDataset(texts, tags, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example of how the data is batched
for batch in dataloader:
    print("Input IDs:", batch["input_ids"])
    print("Attention Mask:", batch["attention_mask"])
    print("Tag IDs:", batch["tag_ids"])