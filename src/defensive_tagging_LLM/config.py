PROMPTS_FOLDER = "./prompts/"
PROMPTS_FILE = PROMPTS_FOLDER + "prompts.json"

DATA_FOLDER = "../../data/"
MSRPC_DATA_FOLDER = DATA_FOLDER + "msrpc/"

QUERY_TAG_IDX = 0
DATA_TAG_IDX = 1

DUP_DETECTION = "Duplicate sentence detection"

# Llama
LLAMA_7B_MODEL_NAME = "meta-llama/llama-2-7b-chat-hf"
LLAMA_3P1_8B_MODEL_NAME = "meta-llama/Llama-3.1-8B"
LLAMA_3P2_3B_MODEL_NAME = "meta-llama/Llama-3.2-3B"
LLAMA_3P2_1B_MODEL_NAME = "meta-llama/Llama-3.2-1B"

def get_data_path(task_name: str, train: bool):
    if (task_name == DUP_DETECTION):
        if train:
            return MSRPC_DATA_FOLDER + "msr_paraphrase_train.txt"
        else:
            return MSRPC_DATA_FOLDER + "msr_paraphrase_test.txt"