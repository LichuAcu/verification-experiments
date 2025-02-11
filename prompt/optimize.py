"""
Optimizes the prompt used to judge the quality of GPQA samples
GPQA with 1000 samples is taken from hazyresearch/GPQA_GPT-4o-mini_v2
"""

import dspy
from dspy import Example
from datasets import load_dataset
from signatures import JudgeSig
import json
import os

NUM_PROBLEMS_TRAIN = 20
NUM_PROBLEMS_VAL = 10
NUM_SAMPLES_PER_PROBLEM = 10

with open('../api-keys.json') as f:
    api_keys = json.load(f)
    openai_key = api_keys['OPENAI_API_KEY'][1]
    os.environ["OPENAI_API_KEY"] = openai_key


def flatten(dataset):
    return [
        Example(
            problem=row["problem"],
            solution_to_judge=row["samples"][i],
            answer_correct=row["answer_correct"][i],
            correct_answer=row["correct_answer"],
        ).with_inputs("problem", "solution_to_judge")
        for row in dataset
        for i in range(NUM_SAMPLES_PER_PROBLEM)
    ]


unflattened_trainset = load_dataset(
    "hazyresearch/GPQA_GPT-4o-mini_v2", split="train").select(range(NUM_PROBLEMS_TRAIN))
trainset = flatten(unflattened_trainset)

unflattened_valset = load_dataset(
    "hazyresearch/GPQA_GPT-4o-mini_v2", split="train").select(range(NUM_PROBLEMS_TRAIN, NUM_PROBLEMS_TRAIN + NUM_PROBLEMS_VAL))
valset = flatten(unflattened_valset)

lm = dspy.LM(
    model="openai/gpt-4o",
)
dspy.settings.configure(lm=lm)


class Judge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(JudgeSig)

    def forward(self, problem, solution_to_judge):
        return self.prog(problem=problem, solution_to_judge=solution_to_judge)


def metric(example, pred, *args, **kwargs):
    verdict = pred["verdict"] == 'True'
    reward = 1 if verdict == example["answer_correct"] else 0
    return reward


judge = Judge()

optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=8)

optimized = optimizer.compile(
    judge, trainset=trainset, valset=valset, requires_permission_to_run=False)

optimized.save("optimized_judge.json")
