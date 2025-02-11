import os
import pandas as pd
from utils.utils import load_api_keys, get_rate_limits
from judges.utils import create_evaluation_prompt, calculate_metrics
from datasets import load_dataset, Dataset
from bespokelabs import curator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", help="Models to use for judging")
args = parser.parse_args()

models = args.models
model_keys, rpm, tpm = get_rate_limits()

os.environ["CURATOR_DISABLE_CACHE"] = "1"
load_api_keys()

NUM_PROBLEMS = 100
NUM_SAMPLES = 10
NUM_JUDGES = 1

unflattened_dataset = load_dataset(
    "hazyresearch/GPQA_GPT-4o-mini_v2", split="train"
).select(range(NUM_PROBLEMS))

dataset = [
    {
        "problem": row["problem"],
        "sample": row["samples"][i],
        "answer_correct": row["answer_correct"][i],
        "correct_answer": row["correct_answer"],
        "problem_idx": problem_idx,
        "sample_idx": i,
        "judge_idx": j,
    }
    for problem_idx, row in enumerate(unflattened_dataset)
    for i in range(NUM_SAMPLES)
    for j in range(NUM_JUDGES)
]

dataset = Dataset.from_list(dataset)

for model_key in models:
    print(f"Judging {model_key}...")
    model = model_keys[model_key]
    max_rpm = rpm[model_key] if model_key in rpm else 4_000
    max_tpm = tpm[model_key] if model_key in tpm else 1_500_000

    llm = curator.LLM(
        model_name=model,
        max_requests_per_minute=max_rpm,
        max_tokens_per_minute=max_tpm,
        prompt_func=create_evaluation_prompt,
        parse_func=lambda row, response: {
            "verdict": (
                (response.lower().rfind("true") > response.lower().rfind("false"))
                if ("true" in response.lower() or "false" in response.lower())
                else False
            ),
            "answer_correct": row["answer_correct"],
            "problem_idx": row["problem_idx"],
            "sample_idx": row["sample_idx"],
            "judge_idx": row["judge_idx"],
        },
        require_all_responses=False,
        max_retries=3,
    )

    judges_dataset = llm(dataset)
    judges_dataset.save_to_disk(f"judges/datasets/{model_key}")

    judges_dataset = pd.DataFrame(judges_dataset)
    metrics = calculate_metrics(judges_dataset)
    print(f"\nMetrics for {model_key}:")
    print(metrics)

    with open(f"judges/stats.txt", "a") as f:
        f.write(f"\n{model_key}:\n")
        f.write(f"num problems: {NUM_PROBLEMS}\n")
        f.write(f"num samples: {NUM_SAMPLES}\n")
        f.write(f"num judges: {NUM_JUDGES}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.3f}\n")
