import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer

class DefenseTagEncoder(nn.Module):
    def __init__(self, num_tags: int, tag_dim: int):
        super().__init__()
        self.tag_embeddings = nn.Embedding(num_tags, tag_dim)

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
        self.llama = LlamaForCausalLM.from_pretrained(llama_model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)

        # DefenseTagEncoder
        self.defense_tag_encoder = DefenseTagEncoder(num_tags=num_tags, tag_dim=self.llama.model.embed_tokens.embedding_dim)

    def forward(self, input_ids, attention_mask, tag_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            tag_ids: (batch_size, seq_len)
        """

        # Standard token embeddings
        token_embeds = self.llama.model.embed_tokens(input_ids)

        # Defense tag embeddings
        tag_embeds = self.defense_tag_encoder(tag_ids)

        # Add defense tag embeddings to token embeddings
        inputs_embeds = token_embeds + tag_embeds

        # 4. Pass through LLaMA
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return outputs
