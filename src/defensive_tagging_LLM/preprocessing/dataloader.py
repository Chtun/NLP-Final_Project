from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

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

def make_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        tag_ids = [item["tag_ids"] for item in batch]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        tag_ids_padded = pad_sequence(tag_ids, batch_first=True, padding_value=0)

        print(tag_ids)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "tag_ids": tag_ids_padded
        }

    return collate_fn
