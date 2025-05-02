from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class TaggingTrainDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples  # List of (texts, tags, outputs) tuples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        texts, tags, outputs = self.examples[idx]

        input_ids, attention_mask, tag_ids = self.tokenize_prompt_segments(texts, tags)

        # Tokenize the expected output
        output_tokens = self.tokenizer(
            outputs,
            add_special_tokens=False,
            return_tensors="pt"
        )
        output_ids = output_tokens.input_ids.squeeze(0)

        # Model training input expects: prompt + expected output
        full_input_ids = torch.cat([input_ids, output_ids], dim=0)
        full_attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(output_ids)
        ], dim=0)

        full_tag_ids = torch.cat([
            tag_ids,
            torch.full((output_ids.size(0),), -1, dtype=torch.long)  # Use -1 to mask response tagging.
        ], dim=0)

        # Labels: same as full_input_ids, but mask prompt tokens with -100
        labels = torch.cat([
            torch.full_like(input_ids, -100),
            output_ids
        ], dim=0)

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "tag_ids": full_tag_ids,
            "labels": labels
        }

    def tokenize_prompt_segments(self, texts, tags):
        """
        Tokenize and concatenate a sequence of text-tag pairs into input_ids, attention_mask, and tag_ids.

        Args:
            texts (List[str]): A list of text segments.
            tags (List[int]): A list of tag values corresponding to each text segment.

        Returns:
            input_ids (torch.LongTensor): Concatenated input token IDs.
            attention_mask (torch.LongTensor): Concatenated attention mask.
            tag_ids (torch.LongTensor): Concatenated tag IDs aligned with input_ids.
        """
        input_ids = []
        attention_mask = []
        tag_ids = []

        for text, tag in zip(texts, tags):
            tokens = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            n_tokens = tokens.input_ids.size(1)
            input_ids.append(tokens.input_ids.squeeze(0))
            attention_mask.append(tokens.attention_mask.squeeze(0))
            tag_ids.append(torch.full((n_tokens,), tag, dtype=torch.long))

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        tag_ids = torch.cat(tag_ids, dim=0)

        return input_ids, attention_mask, tag_ids


    def make_collate_fn(tokenizer):
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            tag_ids = [item["tag_ids"] for item in batch]
            labels = [item["labels"] for item in batch]

            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            tag_ids_padded = pad_sequence(tag_ids, batch_first=True, padding_value=-1)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

            tag_mask = (tag_ids_padded != -1).float() # Tag mask will calculate where to remove the vectors.
            tag_ids_padded[tag_ids_padded == -1] = 0 # Set the values to 0 so that it can be passed through the embedding layer.

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "tag_ids": tag_ids_padded,
                "tag_mask": tag_mask,
                "labels": labels_padded
            }

        return collate_fn
