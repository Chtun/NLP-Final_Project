from torch.utils.data import Dataset
from defensive_tagging_LLM.preprocessing.tagging_train_dataset import TaggingTrainDataset

class TaggingEvalDataset(TaggingTrainDataset):
    def __init__(self, examples, tokenizer, attacked: bool):
        """
        Args:
            examples: List of (texts, tags, expected_output) tuples
            tokenizer: HuggingFace tokenizer
        """
        super().__init__(examples, tokenizer)

        self.attacked = attacked

    def __getitem__(self, idx):

        if self.attacked:
            texts, tags, expected_original_output, expected_attacked_output = self.examples[idx]
        else:
            texts, tags, expected_original_output = self.examples[idx]

        input_ids, attention_mask, tag_ids = self.tokenize_prompt_segments(texts, tags)

        # Tokenize the expected original output
        original_output_tokens = self.tokenizer(
            expected_original_output,
            add_special_tokens=False,
            return_tensors="pt"
        )
        original_output_ids = original_output_tokens.input_ids.squeeze(0)

        if self.attacked:
            # Tokenize the expected attacked output
            attacked_output_tokens = self.tokenizer(
                expected_attacked_output,
                add_special_tokens=False,
                return_tensors="pt"
            )
            attacked_output_ids = attacked_output_tokens.input_ids.squeeze(0)

            data_item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "tag_ids": tag_ids,
                "expected_original_output_ids": original_output_ids,
                "expected_attacked_output_ids": attacked_output_ids
            }
        else:
            data_item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "tag_ids": tag_ids,
                "expected_original_output_ids": original_output_ids
            }



        return data_item
