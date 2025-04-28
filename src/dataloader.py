from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class TaggingDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples  # List of (texts, tags) tuples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        texts, tags = self.examples[idx]

        input_ids = []
        attention_mask = []
        tag_ids = []

        for text, tag in zip(texts, tags):
            # Tokenize individual text without special tokens
            tokens = self.tokenizer(
                text,
                add_special_tokens=False,  # Prevent adding [BOS] or [EOS] here
                return_tensors="pt"
            )

            n_tokens = tokens.input_ids.size(1)  # Number of tokens

            input_ids.append(tokens.input_ids.squeeze(0))
            attention_mask.append(tokens.attention_mask.squeeze(0))
            tag_ids.append(torch.full((n_tokens,), tag, dtype=torch.long))  # Fill n_tokens with the tag

        # Concatenate all parts
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        tag_ids = torch.cat(tag_ids, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tag_ids": tag_ids
        }

# Example: a list of examples, each made up of multiple texts + tags
examples = [
    (["This is a query.", "And this is data."], [0, 1]),
    (["Another query.", "Another piece of data."], [0, 1])
]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# Create dataset and dataloader
dataset = TaggingDataset(examples, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)  # Custom collate_fn since different lengths


for batch in dataloader:
    for item in batch:
        print("Input IDs:", item["input_ids"])
        print("Attention Mask:", item["attention_mask"])
        print("Tag IDs:", item["tag_ids"])
        print()