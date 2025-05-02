import torch.nn as nn
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import os

class DefenseTagEncoder(nn.Module):
    def __init__(self, num_tags: int, tag_dim: int):
        super().__init__()
        self.tag_embeddings = nn.Embedding(num_tags, tag_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the embedding weights with a normal distribution"""
        nn.init.normal_(self.tag_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, tag_indices):
        """
        Args:
            tag_indices: Tensor of shape (batch_size, seq_len) containing tag IDs (0 for DATA, 1 for QUERY)
        Returns:
            tag_embeds: Tensor of shape (batch_size, seq_len, tag_dim)
        """
        return self.tag_embeddings(tag_indices)

class LlamaWithDefenseTags(nn.Module):
    def __init__(self, llama_model_name: str, num_tags: int):
        super().__init__()
        # Load pre-trained LLaMA model
        self.llama = LlamaForCausalLM.from_pretrained(
            llama_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

        # DefenseTagEncoder
        self.defense_tag_encoder = DefenseTagEncoder(num_tags=num_tags, tag_dim=self.llama.model.embed_tokens.embedding_dim)

    def forward(self, input_ids, attention_mask, tag_ids, tag_mask, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            tag_ids: (batch_size, seq_len)
        """

        # Standard token embeddings
        token_embeds = self.llama.model.embed_tokens(input_ids)

        # Defense tag embeddings (learned embeddings for each tag)
        tag_embeds = self.defense_tag_encoder(tag_ids)

        # Apply the tag mask: Set embeddings to 0 where tag_mask is 0 (invalid tags)
        tag_embeds = tag_embeds.masked_fill(tag_mask.unsqueeze(-1) == 0, 0.0)

        assert tag_embeds.size(1) == token_embeds.size(1), "Mismatched sequence lengths after padding"

        # Add defense tag embeddings to token embeddings
        inputs_embeds = token_embeds + tag_embeds

        # Pass through LLaMA
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tag_ids: torch.Tensor,
        tag_mask: torch.Tensor,
        max_new_tokens: int = 100,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate output token IDs using LLaMA with defense tags.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len)
            tag_ids: (batch_size, seq_len)
            tag_mask: (batch_size, seq_len) - 1 for valid tags, 0 for padding
            max_new_tokens: int - how many tokens to generate
            generate_kwargs: other kwargs for `model.generate`

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens) token IDs
        """
        self.eval()
        with torch.no_grad():
            # Get token and tag embeddings
            token_embeds = self.llama.model.embed_tokens(input_ids)
            tag_embeds = self.defense_tag_encoder(tag_ids)
            tag_embeds = tag_embeds.masked_fill(tag_mask.unsqueeze(-1) == 0, 0.0)

            # Add token and tag embeddings
            inputs_embeds = token_embeds + tag_embeds

            # Run generation
            generated_ids = self.llama.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )

        return generated_ids

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        # Save base LLaMA model and tokenizer
        self.llama.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save DefenseTagEncoder weights
        torch.save(self.defense_tag_encoder.state_dict(), os.path.join(save_directory, "defense_tag_encoder.pt"))

        # Save a config for reconstruction (just num_tags and llama_model_name)
        config = {
            "num_tags": self.defense_tag_encoder.tag_embeddings.num_embeddings,  # num_tags
            "llama_model_name": self.llama.config._name_or_path,  # LLaMA model name
        }
        torch.save(config, os.path.join(save_directory, "custom_model_config.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        # Load config
        config = torch.load(os.path.join(load_directory, "custom_model_config.pt"))
        num_tags = config["num_tags"]
        llama_model_name = config["llama_model_name"]

        # Initialize model with the same llama model and num_tags
        model = cls(llama_model_name=llama_model_name, num_tags=num_tags)

        # Load tag encoder weights
        tag_encoder_path = os.path.join(load_directory, "defense_tag_encoder.pt")
        model.defense_tag_encoder.load_state_dict(torch.load(tag_encoder_path))

        return model