import torch

def wrap_prompt_for_llama(user_prompt: str, system_prompt: str = None, include_tools: bool = False) -> str:
    """
    Wrap a user prompt using the Ollama-compatible template for chat models like LLaMA 3.

    Args:
        user_prompt (str): The actual prompt you want the model to answer.
        system_prompt (str): Optional system-level prompt. Defaults to None.
        include_tools (bool): Set to True if you are doing tool-augmented prompting. Defaults to False.

    Returns:
        str: A fully wrapped prompt ready to be tokenized and passed to the model.
    """
    wrapped = ""

    # System prompt (static template)
    wrapped += "<|start_header_id|>system<|end_header_id|>\n"
    wrapped += "Cutting Knowledge Date: December 2023\n\n"
    if system_prompt:
        wrapped += f"{system_prompt}\n"
    if include_tools:
        wrapped += (
            "When you receive a tool call response, use the output to format an answer to the orginal user question.\n\n"
            "You are a helpful assistant with tool calling capabilities.\n"
        )
    wrapped += "<|eot_id|>\n"

    # User prompt
    wrapped += "<|start_header_id|>user<|end_header_id|>\n"
    wrapped += f"{user_prompt}\n"
    wrapped += "<|eot_id|>\n"

    # Assistant prompt (where generation begins)
    wrapped += "<|start_header_id|>assistant<|end_header_id|>\n"

    return wrapped

def wrap_ollama_prompt_tensor(user_prompt_tensor: torch.Tensor, tokenizer, system_prompt_tensor: torch.Tensor=None):
    """
    Wrap tokenized user prompts (batched) with optional system prompt using Ollama-style special token headers.

    Args:
        user_prompt_tensor (torch.Tensor): 2D tensor of tokenized user inputs, shape (k, N).
        tokenizer: Huggingface tokenizer with special tokens defined.
        system_prompt_tensor (torch.Tensor, optional): 1D tensor of system input tokens.

    Returns:
        input_ids (torch.Tensor): Tensor of shape (k, L) where L is max length of wrapped input.
        attention_mask (torch.Tensor): Tensor of shape (k, L), with 1s for real tokens and 0s for padding.
    """
    # Special token IDs
    start_header = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    assert all(tok != tokenizer.unk_token_id for tok in [start_header, end_header, eot]), \
        "Tokenizer does not recognize one or more Ollama special tokens."

    # Encode headers
    system_header = tokenizer.encode("system", add_special_tokens=False)
    user_header = tokenizer.encode("user", add_special_tokens=False)
    assistant_header = tokenizer.encode("assistant", add_special_tokens=False)

    # Use default system prompt if not given
    if system_prompt_tensor is None:
        system_prompt_ids = tokenizer.encode(
            "Cutting Knowledge Date: December 2023\nYou are a helpful assistant.",
            add_special_tokens=False
        )
    else:
        system_prompt_ids = system_prompt_tensor.tolist()

    

    # Build system chunk once
    system_ids = [start_header] + system_header + [end_header] + system_prompt_ids + [eot]
    assistant_ids = [start_header] + assistant_header + [end_header]

    # Build batch
    wrapped_batches = []
    for i in range(user_prompt_tensor.shape[0]):
        user_ids = user_prompt_tensor[i].tolist()
        wrapped = (
            system_ids
            + [start_header] + user_header + [end_header] + user_ids + [eot]
            + assistant_ids
        )
        wrapped_batches.append(wrapped)

    # Return the wrapped input IDs and attention mask without padding
    input_ids = torch.tensor(wrapped_batches, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
