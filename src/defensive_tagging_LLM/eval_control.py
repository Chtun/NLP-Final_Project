from defensive_tagging_LLM.preprocessing.data_preprocessing import *
from defensive_tagging_LLM.preprocessing.injection_preprocessing import *
from defensive_tagging_LLM.preprocessing.tagging_eval_dataset import *
from defensive_tagging_LLM.config import *
from defensive_tagging_LLM.model.model import *
from defensive_tagging_LLM.model.model_utils import *

from transformers import AutoTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
import matplotlib.pyplot as plt
import json


base_model_name = LLAMA_3P2_1B_INSTRUCT_MODEL_NAME # The base LLM's name.
save_folder = "../../output/" # Output folder
num_normal_examples = NORMAL_EXAMPLES_PER_DATASET
num_normal_examples = 10

prompts_dict = extract_prompts(prompt_file=PROMPTS_FILE)

task_names = [
    DUP_DETECTION,
    # GRAMMAR_CORRECTION,
    NAT_LANG_INFERENCE,
    SENT_ANALYSIS,
    SPAM_DETECTION,
    # SUMMARIZATION
]

no_attack_tasks = {}
injected_attack_tasks = {}

corpus_dict = {}

# First, load each corpus
for task_name_i in task_names:
    train_i_file_path = get_data_path(task_name_i, train=True)
    corpus_dict[task_name_i] = load_corpus(task_name_i, train_i_file_path)


for task_name_i in task_names:
    # Generate the No Attack tasks.
    instruction_prompt = prompts_dict[task_name_i]["Instruction prompt"]
    instruction_eval_parsed_corpus = corpus_dict[task_name_i]

    # Sample only up to a set of the corpus to balance each dataset.
    sampled_instruction_corpus = instruction_eval_parsed_corpus[:num_normal_examples]
    processed_instruction_eval = process_tasks(
        task_name=task_name_i,
        input_output=sampled_instruction_corpus,
        prompt=instruction_prompt
    )

    no_attack_tasks[task_name_i] = processed_instruction_eval
    injected_attack_tasks[task_name_i] = {}

    print()
    print(f"Example of no-attack task for {task_name_i}")
    print(processed_instruction_eval[:3])
    print()

    # Generate the Injected Attack tasks.
    for task_name_j in task_names:

        injected_prompt = prompts_dict[task_name_j]["Injected instruction"]

        injected_eval_parsed_corpus = corpus_dict[task_name_j]

        generated_injection_list, expected_attacker_outputs, sampled_target_tasks = generate_target_injection_pairs(
            target_task_name=task_name_i,
            target_task_corpus=instruction_eval_parsed_corpus,
            injected_task_name=task_name_j,
            injected_prompt=injected_prompt,
            injected_task_corpus=injected_eval_parsed_corpus
        )

        processed_injected_tasks = process_tasks_with_injection(
            task_name=task_name_i,
            input_output=sampled_target_tasks,
            prompt=instruction_prompt,
            injection_list=generated_injection_list,
            attacker_output_list=expected_attacker_outputs
        )

        injected_attack_tasks[task_name_i][task_name_j] = processed_injected_tasks

        print(f"Example of injection of task {task_name_j} into original task {task_name_i}:")
        print(processed_injected_tasks[:3])
        print()


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare the data for the custom dataset formatter and loader.
instruction_datasets = {}
injected_datasets = {}

for task_name_i in no_attack_tasks.keys():
    processed_instruction_eval = no_attack_tasks[task_name_i]
    prepared_instruction_eval = [ (row[0], row[1], row[2]) for row in processed_instruction_eval ]

    # Create dataset and dataloader
    instruction_eval_dataset = TaggingEvalDataset(prepared_instruction_eval, tokenizer, attacked=False)

    instruction_datasets[task_name_i] = instruction_eval_dataset

    # for example in instruction_eval_dataset:
    #     print("Input IDs:", example["input_ids"])
    #     print("Attention Mask:", example["attention_mask"])
    #     print("Tag IDs:", example["tag_ids"])
    #     print("Expected Original Outputs:", example["expected_original_output_ids"])
    #     print()
    #     break

for task_name_i in injected_attack_tasks.keys():
    injected_datasets[task_name_i] = {}

    for task_name_j in injected_attack_tasks[task_name_i].keys():
        processed_injected_eval = injected_attack_tasks[task_name_i][task_name_j]
        prepared_injected_eval = [(row[0], row[1], row[2], row[3]) for row in processed_injected_eval ]

        injected_eval_dataset = TaggingEvalDataset(prepared_injected_eval, tokenizer, attacked=True)

        injected_datasets[task_name_i][task_name_j] = injected_eval_dataset

        # print(f"{task_name_i} original x {task_name_j} injected instruction:")
        # for example in injected_eval_dataset:
        #     print("Input IDs:", example["input_ids"])
        #     print("Attention Mask:", example["attention_mask"])
        #     print("Tag IDs:", example["tag_ids"])
        #     print("Expected Original Outputs:", example["expected_original_output_ids"])
        #     print("Expected Attacker Outputs:", example["expected_attacked_output_ids"])
        #     print()
        #     break

# Combine the no-attack and attack-injected into separate datasets for full evaluation
combined_no_attack_datasets = []
combined_attacked_datasets = []

# Combine instruction datasets
for dataset in instruction_datasets.values():
    combined_no_attack_datasets.append(dataset)

# Combine injected datasets
for original_task in injected_datasets.keys():
    for dataset in injected_datasets[original_task].values():
        combined_attacked_datasets.append(dataset)


no_attack_eval_dataset = ConcatDataset(combined_no_attack_datasets)
attacked_eval_dataset = ConcatDataset(combined_attacked_datasets)

# Set the evaluation dataloader
no_attack_eval_dataloader = DataLoader(
    no_attack_eval_dataset,
    batch_size=1,  # One example at a time
    shuffle=True,
    collate_fn=None
)

attacked_eval_dataloader = DataLoader(
    attacked_eval_dataset,
    batch_size=1,  # One example at a time
    shuffle=True,
    collate_fn=None
)

# Calculate the number of batches
num_no_attack_examples = len(no_attack_eval_dataloader)
num_attacked_examples = len(attacked_eval_dataloader)

# Print the number of batches
print(f"Number of no-attack examples: {num_no_attack_examples}")
print(f"Number of attacked examples: {num_attacked_examples}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the saved model
model = LlamaForCausalLM.from_pretrained(
            base_model_name
        )
model = model.to(device)

# Initialize metrics or results storage
no_attack_generation_results = []

# Evaluation loop for no-attack tasks
model.eval()
with torch.no_grad():
    loop = tqdm(no_attack_eval_dataloader, desc="Evaluating")
    for example in loop:
        # Get inputs for evaluation
        input_ids = example["input_ids"].to(device)
        attention_mask = example["attention_mask"].to(device)
        tag_ids = example["tag_ids"].to(device)
        tag_mask = example["tag_mask"].to(device)
        expected_output_ids = example["expected_original_output_ids"].to(device)

        full_formatted_input_ids, full_formatted_attention_mask = wrap_ollama_prompt_tensor(input_ids, tokenizer=tokenizer)
        full_formatted_input_ids = full_formatted_input_ids.to(device)
        full_formatted_attention_mask = full_formatted_attention_mask.to(device)

        # Generate output from the model
        generated_ids = model.generate(
            input_ids=full_formatted_input_ids,
            attention_mask=full_formatted_attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
        )

        new_tokens = generated_ids[:, full_formatted_input_ids.shape[1]:]


        # Decode the generated ids into text
        generated_text = tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=False)

        # Decode the expected output for comparison
        expected_output_text = tokenizer.decode(expected_output_ids.squeeze(0), skip_special_tokens=False)

        # Compare the generated text with the expected output text
        correct = generated_text.strip() == expected_output_text.strip()

        # Store the result
        no_attack_generation_results.append({
            "input_text": tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=False),
            "formatted_input_text": tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=False),
            "expected_output": expected_output_text,
            "generated_output": generated_text,
            "correct": correct
        })

        # Update the loop description with accuracy or any other metrics
        loop.set_postfix(correct=correct)


# After the loop, you can calculate overall accuracy or any other metric
correct_count = sum(result["correct"] for result in no_attack_generation_results)
total_count = len(no_attack_generation_results)
accuracy = correct_count / total_count if total_count > 0 else 0.0

print(f"No-attack Generation Accuracy: {accuracy:.4f}")

os.makedirs(save_folder, exist_ok=True)

with open(f"{save_folder}no_attack_generation_results.json", "w") as f:
    json.dump(no_attack_generation_results, f, indent=4)

# Initialize metrics or results storage
attacked_generation_results = []

# Evaluation loop for no-attack tasks
model.eval()
with torch.no_grad():
    loop = tqdm(attacked_eval_dataloader, desc="Evaluating")
    for example in loop:
        # Get inputs for evaluation
        input_ids = example["input_ids"].to(device)
        attention_mask = example["attention_mask"].to(device)
        tag_ids = example["tag_ids"].to(device)
        tag_mask = example["tag_mask"].to(device)
        expected_original_output_ids = example["expected_original_output_ids"].to(device)
        expected_attacker_output_ids = example["expected_attacked_output_ids"].to(device)

        full_formatted_input_ids, full_formatted_attention_mask = wrap_ollama_prompt_tensor(input_ids, tokenizer=tokenizer)
        full_formatted_input_ids = full_formatted_input_ids.to(device)
        full_formatted_attention_mask = full_formatted_attention_mask.to(device)


        # Generate output from the model
        generated_ids = model.generate(
            input_ids=full_formatted_input_ids,
            attention_mask=full_formatted_attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode the generated ids into text
        generated_text = tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)

        # Decode the expected output for comparison
        expected_original_output_text = tokenizer.decode(expected_original_output_ids.squeeze(0), skip_special_tokens=True)
        expected_attacker_output_text = tokenizer.decode(expected_attacker_output_ids.squeeze(0), skip_special_tokens=True)

        # Compare the generated text with the expected output text
        correct = generated_text.strip() == expected_original_output_text.strip()
        attack_success = generated_text.strip() == expected_attacker_output_text.strip()

        # Store the result
        attacked_generation_results.append({
            "input_text": tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True),
            "fully_formatted_input": tokenizer.decode(full_formatted_input_ids.squeeze(0), skip_special_tokens=True),
            "expected_original_output": expected_original_output_text,
            "expected_attacked_output": expected_attacker_output_text,
            "generated_output": generated_text,
            "correct": correct,
            "attack_success": attack_success
        })

        # Update the loop description with accuracy or any other metrics
        loop.set_postfix(correct=correct)


# After the loop, you can calculate overall accuracy or any other metric
correct_count = sum(result["correct"] for result in attacked_generation_results)
total_count = len(attacked_generation_results)
accuracy = correct_count / total_count if total_count > 0 else 0.0
attack_success_count = sum(result["attack_success"] for result in attacked_generation_results)
attack_success_rate = attack_success_count / total_count if total_count > 0 else 0.0

print(f"Attacked Generation Accuracy: {accuracy:.4f}")
print(f"Attack Success Rate: {attack_success_rate:.4f}")

with open(f"{save_folder}attacked_generation_results.json", "w") as f:
    json.dump(no_attack_generation_results, f, indent=4)
