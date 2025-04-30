import torch.nn as nn
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

class DefenseTagEncoder(nn.Module):
    def __init__(self, num_tags: int, tag_dim: int):
        super().__init__()
        self.tag_embeddings = nn.Embedding(num_tags, tag_dim).half()
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
            llama_model_name,
            torch_dtype=torch.float16
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

        # Ensure tag_embeds and token_embeds have the same length (pad or truncate tag_embeds if needed)
        if tag_embeds.size(1) < token_embeds.size(1):
            # Pad tag_embeds with zeros if tag_embeds is shorter than token_embeds
            pad_size = token_embeds.size(1) - tag_embeds.size(1)
            tag_embeds = torch.cat([tag_embeds, torch.zeros(tag_embeds.size(0), pad_size, tag_embeds.size(2), device=tag_embeds.device)], dim=1)
        elif tag_embeds.size(1) > token_embeds.size(1):
            # Truncate tag_embeds if it's longer than token_embeds
            tag_embeds = tag_embeds[:, :token_embeds.size(1), :]

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
