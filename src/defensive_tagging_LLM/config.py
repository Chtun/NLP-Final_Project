# Prompt folder
PROMPTS_FOLDER = "../prompts/"
PROMPTS_FILE = PROMPTS_FOLDER + "prompts.json"

# Data folders
DATA_FOLDER = "../../data/"
MSRPC_DATA_FOLDER = DATA_FOLDER + "msrpc/"
GIGAWORD_DATA_FOLDER = DATA_FOLDER + "gigaword/"
JFLEG_DATA_FOLDER = DATA_FOLDER + "jfleg/"
SST2_DATA_FOLDER = DATA_FOLDER + "sst2/"
SMS_SPAM_DATA_FOLDER = DATA_FOLDER + "sms-spam/"
GLUE_DATA_FOLDER = DATA_FOLDER + "glue/"

# Model weights folder
MODEL_WEIGHTS_FOLDER = "../../output/model_weights"

# Tag indices
QUERY_TAG_IDX = 0
DATA_TAG_IDX = 1

# Task names
DUP_DETECTION = "Duplicate sentence detection"
GRAMMAR_CORRECTION = "Grammar correction"
NAT_LANG_INFERENCE = "Natural language inference"
SENT_ANALYSIS = "Sentiment analysis"
SPAM_DETECTION = "Spam detection"
SUMMARIZATION = "Summarization"

# Llama
LLAMA_7B_MODEL_NAME = "meta-llama/llama-2-7b-chat-hf"
LLAMA_3P1_8B_MODEL_NAME = "meta-llama/Llama-3.1-8B"
LLAMA_3P2_3B_MODEL_NAME = "meta-llama/Llama-3.2-3B"
LLAMA_3P2_1B_MODEL_NAME = "meta-llama/Llama-3.2-1B"
LLAMA_3P2_1B_INSTRUCT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Training configs.
NORMAL_EXAMPLES_PER_DATASET = 1500
INJECTED_EXAMPLES_PER_CROSS = 200


# Grabbing data path name.
def get_data_path(task_name: str, train: bool):
    if task_name == DUP_DETECTION:
        if train:
            return MSRPC_DATA_FOLDER + "msr_paraphrase_train.txt"
        else:
            return MSRPC_DATA_FOLDER + "msr_paraphrase_test.txt"
    elif task_name == GRAMMAR_CORRECTION:
        if train:
            return JFLEG_DATA_FOLDER + "train.csv"
        else:
            return JFLEG_DATA_FOLDER + "eval.csv"
    elif task_name == NAT_LANG_INFERENCE:
        if train:
            return GLUE_DATA_FOLDER + "train.csv"
        else:
            return GLUE_DATA_FOLDER + "validation_matched.csv"
    elif task_name == SENT_ANALYSIS:
        if train:
            return SST2_DATA_FOLDER + "train.csv"
        else:
            return SST2_DATA_FOLDER + "validation.csv"
    elif task_name == SPAM_DETECTION:
        return SMS_SPAM_DATA_FOLDER + "spam.csv"
    elif task_name == SUMMARIZATION:
        return GIGAWORD_DATA_FOLDER + "gigaData.csv"