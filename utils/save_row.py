from datasets import load_dataset
import json

dataset = load_dataset("hazyresearch/GPQA_GPT-4o-mini_v2", split="train")

with open("temp/single_row.json", "w") as f:
    json.dump(dataset[0], f)