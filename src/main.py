import json
from config import MSRPC_DATA_FOLDER, PROMPTS_FILE
from preprocessing.data_preprocessing import load_msr_dup_detection_corpus, process_dup_detection, process_dup_detection_with_injection, generate_injection_task_strs
from preprocessing.injection_preprocessing import generate_injection_list, get_injection_prepend
import random

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from preprocessing.dataloader import TaggingDataset
from config import QUERY_TAG_IDX, DATA_TAG_IDX


with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts_data = json.load(f)

# Convert the JSON into a dictionary of dictionaries
prompts_dict = {}

for task in prompts_data:
    task_name = task['task']
    task_prompts = {prompt['type']: prompt['text'] for prompt in task['prompts']}
    prompts_dict[task_name] = task_prompts


dup_detection_instruction_prompt = prompts_dict["Duplicate sentence detection"]["Instruction prompt"]
dup_detection_injection_prompt = prompts_dict["Duplicate sentence detection"]["Injected instruction"]

dup_detection_test_file_path = MSRPC_DATA_FOLDER + "msr_paraphrase_test.txt"
dup_detection_instruction_test_dataset = load_msr_dup_detection_corpus(dup_detection_test_file_path)
dup_detection_injection_test_dataset = load_msr_dup_detection_corpus(dup_detection_test_file_path)

processed_dup_detection_instruction_test = process_dup_detection(
    dup_detection_tuples=dup_detection_instruction_test_dataset,
    prompt=dup_detection_instruction_prompt
    )

pos_match_test_dataset = [t for t in dup_detection_injection_test_dataset if t[2] == 1]
neg_match_test_dataset = [t for t in dup_detection_injection_test_dataset if t[2] == 0]

# Process the injected tasks into list of strings.
processed_pos_injected_tasks = generate_injection_task_strs(
    injection_set=pos_match_test_dataset,
    injected_prompt=dup_detection_injection_prompt
)
processed_neg_injected_tasks = generate_injection_task_strs(
    injection_set=neg_match_test_dataset,
    injected_prompt=dup_detection_injection_prompt
)

# Generate samples for positive and negative target tasks as well as injection tasks.
sampled_pos_target_set = random.sample(pos_match_test_dataset, min(100, len(pos_match_test_dataset)))
sampled_neg_target_set = random.sample(neg_match_test_dataset, min(100, len(neg_match_test_dataset)))

injected_prepend = get_injection_prepend()

# Generate a list of prompt injections.
pos_injection_list = generate_injection_list(
    injected_prepend=injected_prepend,
    injected_task_list=processed_pos_injected_tasks
)

neg_injection_list = generate_injection_list(
    injected_prepend=injected_prepend,
    injected_task_list=processed_neg_injected_tasks
)

processed_pos_injected_set = process_dup_detection_with_injection(
    dup_detection_tuples=sampled_pos_target_set,
    prompt=dup_detection_instruction_prompt,
    injection_list=neg_injection_list
    )

processed_neg_injected_set = process_dup_detection_with_injection(
    dup_detection_tuples=sampled_neg_target_set,
    prompt=dup_detection_instruction_prompt,
    injection_list=pos_injection_list
    )

processed_dup_detection_injection_test = processed_pos_injected_set + processed_neg_injected_set

# Prepare the data for the custom dataset formatter and loader.
dup_detection_instruction_test_prepared = [ (row[0], row[1]) for row in processed_dup_detection_instruction_test ]
dup_detection_injection_test_prepared = [ (row[0], row[1]) for row in processed_dup_detection_injection_test ]


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# Create dataset and dataloader
dup_detection_instruction_test_dataset = TaggingDataset(dup_detection_instruction_test_prepared, tokenizer)
dup_detection_instruction_test_dataloader = DataLoader(dup_detection_instruction_test_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

dup_detection_injection_test_dataset = TaggingDataset(dup_detection_injection_test_prepared, tokenizer)
dup_detection_injection_test_dataloader = DataLoader(dup_detection_injection_test_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

# print("Duplicate Detection No Attack Instruction:")
# for batch in dup_detection_instruction_test_dataloader:
#     for item in batch:
#         print("Input IDs:", item["input_ids"])
#         print("Attention Mask:", item["attention_mask"])
#         print("Tag IDs:", item["tag_ids"])
#         print()

# print("Duplicate Detection Attacked Instruction:")
# for batch in dup_detection_injection_test_dataloader:
#     for item in batch:
#         print("Input IDs:", item["input_ids"])
#         print("Attention Mask:", item["attention_mask"])
#         print("Tag IDs:", item["tag_ids"])
#         print()

print(len(dup_detection_instruction_test_prepared))
print(len(dup_detection_injection_test_prepared))



