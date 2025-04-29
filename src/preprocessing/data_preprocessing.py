import csv
from config import QUERY_TAG_IDX, DATA_TAG_IDX
import random

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

def process_dup_detection(dup_detection_tuples: list, prompt: str) -> list:
    empty_injection_list = ["" for i in dup_detection_tuples]

    return process_dup_detection_with_injection(dup_detection_tuples, prompt, empty_injection_list)


def process_dup_detection_with_injection(dup_detection_tuples: list, prompt: str, injection_list: list) -> list:
    if (len(dup_detection_tuples) != len(injection_list)):
        raise Exception("The input-output pair list and injection lists are not the same size!")
    
    processed_tuples = []
    
    for index in range(len(dup_detection_tuples)):
        query = prompt
        injection = injection_list[index]
        data = str(dup_detection_tuples[index][0]) + "\n" + str(dup_detection_tuples[index][1]) + injection

        if dup_detection_tuples[index][2] == 1:
            output = "paraphrased"
        else:
            output = "not paraphrased"

        text_set = [query, data]
        tag_set = [QUERY_TAG_IDX, DATA_TAG_IDX]

        processed_tuples.append([text_set, tag_set, output])

    return processed_tuples

def generate_injection_task_strs(injection_set: list, injected_prompt: str, num_samples: int=100):
    # Generate samples for positive and negative target tasks as well as injection tasks.
    sampled_injection_set = random.sample(injection_set, min(num_samples, len(injection_set)))

    # Process the injected tasks.
    processed_injected_set = process_dup_detection(
        dup_detection_tuples=sampled_injection_set,
        prompt=injected_prompt
    )
    processed_injected_set = [
        elem[0][0] + "\n" + elem[0][1] for elem in processed_injected_set
    ]

    return processed_injected_set