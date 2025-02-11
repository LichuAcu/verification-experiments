import json
import os
import re


def extract_answer(sample):
    pattern = r"the answer is:\s*([abcd])"
    matches = re.findall(pattern, sample.lower())
    return matches[-1].upper() if matches else None


def load_api_keys():
    api_keys_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "api-keys.json")
    with open(api_keys_path) as f:
        api_keys = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"][0]
    os.environ["ANTHROPIC_API_KEY"] = api_keys["ANTHROPIC_API_KEY"][0]
    os.environ["TOGETHER_API_KEY"] = api_keys["TOGETHER_API_KEY"][1]
    os.environ["DEEPSEEK_API_KEY"] = api_keys["DEEPSEEK_API_KEY"][0]


def get_rate_limits():
    models = {
        "4o": "gpt-4o",
        "4o-mini": "gpt-4o-mini",
        "Claude3": "claude-3-5-sonnet-20240620",
        "Qwen72B": "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen7B": "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Llama8B": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Llama70B": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Llama405B": "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "R1-1.5B": "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "R1-70B": "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
        "V3": "together_ai/deepseek-ai/DeepSeek-V3",
        "Gemma2-27B": "together_ai/google/gemma-2-27b-it",
        "Gemma2-9B": "together_ai/google/gemma-2-9b-it",
        "Mistral24B": "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",
        "Mistral7B": "together_ai/mistralai/Mistral-7B-Instruct-v0.1",
    }

    rpm = {
        "4o": 5_000,
        "4o-mini": 5_000,
        "Claude3": 3_000,
    }

    tpm = {
        "4o": 10_000_000,
        "4o-mini": 10_000_000,
        "Claude3": 200_000,
    }

    return models, rpm, tpm
