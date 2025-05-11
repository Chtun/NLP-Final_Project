from defensive_tagging_LLM.preprocessing.data_preprocessing import *
from defensive_tagging_LLM.preprocessing.injection_preprocessing import *
from defensive_tagging_LLM.preprocessing.tagging_train_dataset import *
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
    instruction_train_parsed_corpus = corpus_dict[task_name_i]

    # Sample only up to a set of the corpus to balance each dataset.
    sampled_instruction_corpus = instruction_train_parsed_corpus[:NORMAL_EXAMPLES_PER_DATASET]
    processed_instruction_train = process_tasks(
        task_name=task_name_i,
        input_output=sampled_instruction_corpus,
        prompt=instruction_prompt
    )

    no_attack_tasks[task_name_i] = processed_instruction_train
    injected_attack_tasks[task_name_i] = {}

    print()
    print(f"Example of no-attack task for {task_name_i}")
    print(processed_instruction_train[:3])
    print()

    # Generate the Injected Attack tasks.
    for task_name_j in task_names:

        injected_prompt = prompts_dict[task_name_j]["Injected instruction"]

        injected_train_parsed_corpus = corpus_dict[task_name_j]

        generated_injection_list, expected_attacker_outputs, sampled_target_tasks = generate_target_injection_pairs(
            target_task_name=task_name_i,
            target_task_corpus=instruction_train_parsed_corpus,
            injected_task_name=task_name_j,
            injected_prompt=injected_prompt,
            injected_task_corpus=injected_train_parsed_corpus
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
    processed_instruction_train = no_attack_tasks[task_name_i]
    prepared_instruction_train = [ (row[0], row[1], row[2]) for row in processed_instruction_train ]

    # Create dataset and dataloader
    instruction_train_dataset = TaggingTrainDataset(prepared_instruction_train, tokenizer)

    instruction_datasets[task_name_i] = instruction_train_dataset

    # for example in instruction_train_dataset:
    #     print("Input IDs:", example["input_ids"])
    #     print("Attention Mask:", example["attention_mask"])
    #     print("Tag IDs:", example["tag_ids"])
    #     print()
    #     break

for task_name_i in injected_attack_tasks.keys():
    injected_datasets[task_name_i] = {}

    for task_name_j in injected_attack_tasks[task_name_i].keys():
        processed_injected_train = injected_attack_tasks[task_name_i][task_name_j]
        prepared_injected_train = [(row[0], row[1], row[2]) for row in processed_injected_train ]

        injected_train_dataset = TaggingTrainDataset(prepared_injected_train, tokenizer)

        injected_datasets[task_name_i][task_name_j] = injected_train_dataset

        # print(f"{task_name_i} original x {task_name_j} injected instruction:")
        # for example in injected_train_dataset:
        #     print("Input IDs:", example["input_ids"])
        #     print("Attention Mask:", example["attention_mask"])
        #     print("Tag IDs:", example["tag_ids"])
        #     print()
        #     break

# Combine the datasets for full training
combined_datasets = []

# Combine instruction datasets
for dataset in instruction_datasets.values():
    combined_datasets.append(dataset)

# Combine injected datasets
for original_task in injected_datasets.keys():
    for dataset in injected_datasets[original_task].values():
        combined_datasets.append(dataset)


full_train_dataset = ConcatDataset(combined_datasets)

# Set the training dataloader
train_dataloader = instruction_train_dataloader = DataLoader(
    full_train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=TaggingTrainDataset.make_collate_fn(tokenizer)
)

# Calculate the number of batches
num_batches = len(train_dataloader)

# Print the number of batches
print(f"Number of batches: {num_batches}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 4

# # Initialize the model
# model = LlamaWithDefenseTags(llama_model_name=base_model_name, num_tags=2)  # Replace with actual model name
# model = model.to(device)

# # Create the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-7)

# loss_history = []

# # Training loop
# model.train()
# for epoch in range(num_epochs):
#     loop = tqdm(instruction_train_dataloader, desc=f"Epoch {epoch+1}")
#     for batch in loop:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         tag_ids = batch['tag_ids'].to(device)
#         tag_mask = batch['tag_mask'].to(device)
#         labels = batch['labels'].to(device)

#         # with autocast():
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             tag_ids=tag_ids,
#             tag_mask=tag_mask,
#             labels=labels
#         )
#         loss = outputs.loss

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         loop.set_postfix(loss=loss.item())

#         loss_history.append(loss.item())  # Save loss


# # Create a model weights folder, if not made already
# os.makedirs(MODEL_WEIGHTS_FOLDER, exist_ok=True)

# # Push model weights
# model_name = f"{base_model_name}_Defensive_Tagging"
# clean_model_name = re.sub(r'[^\w\-]', '_', model_name)
# model_folder_path = os.path.join(MODEL_WEIGHTS_FOLDER, clean_model_name)
# model.save_pretrained(model_folder_path)

# hf_token = os.getenv("HF_TOKEN")

# api = HfApi(token=hf_token)
# api.upload_folder(
#     folder_path=model_folder_path,
#     repo_id="Chtun/Defensive_Tagging_LLM",  # Your full model path on HF Hub
#     repo_type="model",
# )

# plt.figure(figsize=(10, 5))
# plt.plot(loss_history, label='Training Loss')
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Model Training Loss")
# plt.legend()
# plt.grid(True)
# plt.show()