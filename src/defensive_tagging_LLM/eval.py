from defensive_tagging_LLM.preprocessing.data_preprocessing import *
from defensive_tagging_LLM.preprocessing.injection_preprocessing import *
from defensive_tagging_LLM.preprocessing.tagging_eval_dataset import *
from defensive_tagging_LLM.config import *
from defensive_tagging_LLM.model.model import *

from transformers import AutoTokenizer
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW

from tqdm import tqdm
import matplotlib.pyplot as plt

from huggingface_hub import HfApi


base_model_name = LLAMA_3P2_1B_MODEL_NAME # The base LLM's name.
repo_name = "Chtun/Defensive_Tagging_LLM" # For saving this model to a huggingface repo.

prompts_dict = extract_prompts(prompt_file=PROMPTS_FILE)

task_names = [
    DUP_DETECTION,
    GRAMMAR_CORRECTION,
    NAT_LANG_INFERENCE,
    SENT_ANALYSIS,
    SPAM_DETECTION,
    SUMMARIZATION
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
    sampled_instruction_corpus = instruction_eval_parsed_corpus[:NORMAL_EXAMPLES_PER_DATASET]
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

    for example in instruction_eval_dataset:
        print("Input IDs:", example["input_ids"])
        print("Attention Mask:", example["attention_mask"])
        print("Tag IDs:", example["tag_ids"])
        print("Expected Original Outputs:", example["expected_original_output_ids"])
        print()
        break

for task_name_i in injected_attack_tasks.keys():
    injected_datasets[task_name_i] = {}

    for task_name_j in injected_attack_tasks[task_name_i].keys():
        processed_injected_eval = injected_attack_tasks[task_name_i][task_name_j]
        prepared_injected_eval = [(row[0], row[1], row[2], row[3]) for row in processed_injected_eval ]

        injected_eval_dataset = TaggingEvalDataset(prepared_injected_eval, tokenizer, attacked=True)

        injected_datasets[task_name_i][task_name_j] = injected_eval_dataset

        print(f"{task_name_i} original x {task_name_j} injected instruction:")
        for example in injected_eval_dataset:
            print("Input IDs:", example["input_ids"])
            print("Attention Mask:", example["attention_mask"])
            print("Tag IDs:", example["tag_ids"])
            print("Expected Original Outputs:", example["expected_original_output_ids"])
            print("Expected Attacker Outputs:", example["expected_attacked_output_ids"])
            print()
            break

# Combine the datasets for full evaluation
combined_datasets = []

# Combine instruction datasets
for dataset in instruction_datasets.values():
    combined_datasets.append(dataset)

# Combine injected datasets
for original_task in injected_datasets.keys():
    for dataset in injected_datasets[original_task].values():
        combined_datasets.append(dataset)


full_eval_dataset = ConcatDataset(combined_datasets)

# Set the evaluation dataloader
eval_dataloader = DataLoader(
    full_eval_dataset,
    batch_size=1,  # One example at a time
    shuffle=True,  # Shuffle the dataset (optional, depending on your use case)
    collate_fn=None  # No need for collate function when batch size is 1
)

# Calculate the number of batches
num_batches = len(eval_dataloader)

# Print the number of batches
print(f"Number of batches: {num_batches}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 4

# # Initialize the model
# model = LlamaWithDefenseTags(llama_model_name=base_model_name, num_tags=2)  # Replace with actual model name
# model = model.to(device)

# # Initialize metrics or results storage
# generation_results = []

# # Evaluation loop
# model.eval()  # Switch model to evaluation mode
# with torch.no_grad():  # No gradients are needed for evaluation
#     for epoch in range(num_epochs):
#         loop = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}")
#         for example in loop:
#             # Get inputs for evaluation
#             input_ids = example["input_ids"].to(device)
#             attention_mask = example["attention_mask"].to(device)
#             expected_output_ids = example["expected_original_output_ids"].to(device)

#             # Generate output from the model
#             generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)  # Adjust max_length if needed

#             # Decode the generated ids into text
#             generated_text = model.tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)

#             # Decode the expected output for comparison
#             expected_output_text = model.tokenizer.decode(expected_output_ids.squeeze(0), skip_special_tokens=True)

#             # Compare the generated text with the expected output text
#             correct = generated_text.strip() == expected_output_text.strip()

#             # Store the result
#             generation_results.append({
#                 "input_text": model.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True),
#                 "expected_output": expected_output_text,
#                 "generated_output": generated_text,
#                 "correct": correct
#             })

#             # Update the loop description with accuracy or any other metrics
#             loop.set_postfix(correct=correct)

# # After the loop, you can calculate overall accuracy or any other metric
# correct_count = sum(result["correct"] for result in generation_results)
# total_count = len(generation_results)
# accuracy = correct_count / total_count if total_count > 0 else 0.0

# print(f"Generation Accuracy: {accuracy:.4f}")