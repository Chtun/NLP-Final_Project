import csv
from defensive_tagging_LLM.config import *
import random
import json
from defensive_tagging_LLM.preprocessing.injection_preprocessing import *

def extract_prompts(prompt_file: str) -> dict:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

        # Convert the JSON into a dictionary of dictionaries
        prompts_dict = {}

        for task in prompts_data:
            task_name = task['task']
            task_prompts = {prompt['type']: prompt['text'] for prompt in task['prompts']}
            prompts_dict[task_name] = task_prompts
        
        return prompts_dict

def load_corpus(task_name: str, file_path: str) -> list:
    if task_name == DUP_DETECTION:
        return load_msr_dup_detection_corpus(filepath=file_path)
    else:
        raise Exception("Task name not recognized!")

def load_msr_dup_detection_corpus(filepath: str) -> list:
    pairs = []

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter='\t')  # Tab-separated values

        for row in reader:
            # Each row is a dictionary now
            text1 = row['#1 String']
            text2 = row['#2 String']
            label = int(row['Quality'])  # 1 = paraphrase, 0 = not paraphrase

            pairs.append((text1, text2, label))

    return pairs


def process_tasks(task_name: str, input_output: list, prompt: str) -> list:
    empty_injection_list = ["" for i in input_output]

    return process_tasks_with_injection(task_name, input_output, prompt, empty_injection_list)


def process_tasks_with_injection(task_name: str, input_output: list, prompt: str, injection_list: list) -> list:
    if (len(input_output) != len(injection_list)):
        raise Exception("The input-output pair list and injection lists are not the same size!")
    
    processed_tuples = []
    
    for index in range(len(input_output)):
        query = prompt
        injection = injection_list[index]

        if (task_name == DUP_DETECTION):
            data = str(input_output[index][0]) + "\n" + str(input_output[index][1]) + injection

            if input_output[index][2] == 1:
                output = "paraphrased"
            else:
                output = "not paraphrased"
        else:
            raise Exception("Task name not recognized!")

        text_set = [query, data]
        tag_set = [QUERY_TAG_IDX, DATA_TAG_IDX]

        processed_tuples.append([text_set, tag_set, output])

    return processed_tuples

def generate_injection_task_strs(injected_task_name: str, injection_set: list, injected_prompt: str, num_samples: int=100):
    # Generate samples for positive and negative target tasks as well as injection tasks.
    sampled_injection_set = random.sample(injection_set, min(num_samples, len(injection_set)))

    # Process the injected tasks.
    processed_injected_set = process_tasks(
        task_name=injected_task_name,
        input_output=sampled_injection_set,
        prompt=injected_prompt
    )
    processed_injected_set = [
        elem[0][0] + "\n" + elem[0][1] for elem in processed_injected_set
    ]

    return processed_injected_set

def generate_target_injection_pairs(target_task_name: str, target_task_corpus: list, injected_task_name: str, injected_prompt: str, injected_task_corpus: str, num_samples: int=200) -> tuple[list, list]:
    sampled_injected_tasks = []
    sampled_target_tasks = []

    if target_task_name == injected_task_name:
        print(f"Handling when tasks are the same for: {target_task_name}")
        if target_task_name == DUP_DETECTION:
            # Generate samples for positive and negative target/injection tasks.
            pos_match_train = [t for t in target_task_corpus if t[2] == 1]
            neg_match_train = [t for t in target_task_corpus if t[2] == 0]

            # Process the injected tasks into a sample of list of strings.
            processed_pos_injected_tasks = generate_injection_task_strs(
                injected_task_name=injected_task_name,
                injection_set=pos_match_train,
                injected_prompt=injected_prompt,
                num_samples=int(num_samples/2)
            )
            processed_neg_injected_tasks = generate_injection_task_strs(
                injected_task_name=injected_task_name,
                injection_set=neg_match_train,
                injected_prompt=injected_prompt,
                num_samples=int(num_samples/2)
            )

            sampled_injected_tasks = processed_neg_injected_tasks + processed_pos_injected_tasks

            # Generate samples for positive and negative target tasks.
            sampled_pos_target_tasks = random.sample(pos_match_train, min(int(num_samples/2), len(pos_match_train)))
            sampled_neg_target_tasks = random.sample(neg_match_train, min(int(num_samples/2), len(neg_match_train)))

            sampled_target_tasks = sampled_pos_target_tasks + sampled_neg_target_tasks

        else:
            raise Exception("Task name not recognized!")
    else:
        # Process the injected tasks into a sample of list of strings.
        sampled_injected_tasks = generate_injection_task_strs(
                injected_task_name=injected_task_name,
                injection_set=injected_task_corpus,
                injected_prompt=injected_prompt,
                num_samples=num_samples
            )
        
        # Generate samples for target tasks.
        sampled_target_tasks = random.sample(target_task_corpus, num_samples)


    injected_prepend = get_injection_prepend()

    # Generate a list of prompt injections.
    injection_list = generate_injection_list(
        injected_prepend=injected_prepend,
        injected_task_list=sampled_injected_tasks
    )

    return injection_list, sampled_target_tasks