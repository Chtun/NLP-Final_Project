from defensive_tagging_LLM.preprocessing.data_preprocessing import *
from defensive_tagging_LLM.preprocessing.injection_preprocessing import *
from defensive_tagging_LLM.preprocessing.dataloader import *
from defensive_tagging_LLM.config import *
from defensive_tagging_LLM.model.model import *

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from torch.optim import AdamW
from tqdm import tqdm

prompt_file = "../prompts/" +  "prompts.json"

prompts_dict = extract_prompts(prompt_file=prompt_file)


task_names = [
    "Duplicate sentence detection"
]

no_attack_tasks = {}
injected_attack_tasks = {}


for task_name_i in task_names:
    # Generate the No Attack tasks.
    train_i_file_path = get_data_path(task_name_i, train=True)

    instruction_prompt = prompts_dict[task_name_i]["Instruction prompt"]
    instruction_train_parsed_corpus = load_corpus(task_name_i, train_i_file_path)
    processed_instruction_train = process_tasks(
        task_name=task_name_i,
        input_output=instruction_train_parsed_corpus,
        prompt=instruction_prompt
    )

    no_attack_tasks[task_name_i] = processed_instruction_train
    injected_attack_tasks[task_name_i] = {}

    # Generate the Injected Attack tasks.
    for task_name_j in task_names:
        injected_prompt = prompts_dict[task_name_j]["Injected instruction"]

        train_j_file_path = get_data_path(task_name_j, train=True)
        
        injected_train_parsed_corpus = load_corpus(task_name_j, train_j_file_path)

        generated_injection_list, sampled_target_tasks = generate_target_injection_pairs(
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
            injection_list=generated_injection_list
        )

        injected_attack_tasks[task_name_i][task_name_j] = processed_injected_tasks


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA_7B_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Prepare the data for the custom dataset formatter and loader.
instruction_dataloaders = {}
injected_dataloaders = {}

for task_name_i in no_attack_tasks.keys():
    processed_instruction_train = no_attack_tasks[task_name_i]
    prepared_instruction_train = [ (row[0], row[1], row[2]) for row in processed_instruction_train ]

    # Create dataset and dataloader
    instruction_train_dataset = TaggingDataset(prepared_instruction_train, tokenizer)
    instruction_train_dataloader = DataLoader(
        instruction_train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer)
        )
    
    instruction_dataloaders[task_name_i] = instruction_train_dataloader

    print(f"{task_name_i} No Attack instruction:")
    for batch in instruction_train_dataloader:
        print("Input IDs:", batch["input_ids"])
        print("Attention Mask:", batch["attention_mask"])
        print("Tag IDs:", batch["tag_ids"])
        print()


for task_name_i in injected_attack_tasks.keys():
    injected_dataloaders[task_name_i] = {}

    for task_name_j in injected_attack_tasks[task_name_i].keys():
        processed_injected_train = injected_attack_tasks[task_name_i][task_name_j]
        prepared_injected_train = [(row[0], row[1], row[2]) for row in processed_injected_train ]

        injected_train_dataset = TaggingDataset(prepared_injected_train, tokenizer)
        injected_train_dataloader = DataLoader(
            injected_train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=make_collate_fn(tokenizer)
            )
        
        injected_dataloaders[task_name_i][task_name_j] = injected_train_dataloader

        print(f"{task_name_i} original x {task_name_j} injected instruction:")
        for batch in injected_train_dataloader:
            print("Input IDs:", batch["input_ids"])
            print("Attention Mask:", batch["attention_mask"])
            print("Tag IDs:", batch["tag_ids"])
            print()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 20

# # Initialize the model
# model = LlamaWithDefenseTags(llama_model_name=LLAMA_7B_MODEL_NAME, num_tags=2)  # Replace with actual model name
# model = model.to(device)  # Ensure you're using the correct device (GPU/CPU)

# # Create the optimizer
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # Set your dataloader
# instruction_train_dataloader = instruction_dataloaders[DUP_DETECTION]

# # Training loop
# model.train()
# for epoch in range(num_epochs):
#     loop = tqdm(instruction_train_dataloader, desc=f"Epoch {epoch+1}")
#     for batch in loop:
#         # Move data to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         tag_ids = batch['tag_ids'].to(device)
#         labels = batch['labels'].to(device)

#         # Forward pass: compute logits and loss
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             tag_ids=tag_ids,
#             labels=labels
#         )
#         loss = outputs.loss

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Log loss
#         loop.set_postfix(loss=loss.item())
