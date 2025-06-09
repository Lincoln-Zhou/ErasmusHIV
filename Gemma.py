from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import pipeline, BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
from typing import Optional
import os
import time

from experimental import run_unsloth
from utilities import parse_gemma_output, build_dataset


def run(prompt: str, pipe):
    # Execute a single prompt and get parsed response from LLM

    messages = [
        {"role": "system", "content": "You are a clinical decision support assistant. Your task is to help determine whether HIV testing is recommended for a patient based on Dutch medical notes (EHR text). You follow clinical guidelines and reason step by step before making a decision. Only recommend testing if there is a clear indicator."},
        {"role": "user", "content": prompt}
    ]

    output = pipe(messages, max_new_tokens=2048)
    response = output[0]["generated_text"][-1]["content"]

    decision = parse_gemma_output(response)

    return decision, response


def evaluate(dataset: str | pd.DataFrame, pipe):
    # Execute a prompt dataset and evaluate against the provided labels

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    predictions, outputs = [], []

    for idx, row in tqdm(dataset.iterrows()):
        prompt = row['prompt']

        if not isinstance(pipe, tuple):
            prediction, output = run(prompt, pipe)
        else:
            prediction, output = run_unsloth(prompt, pipe)

        predictions.append(prediction)
        outputs.append(output)

    predictions = np.array(predictions).astype(int)

    # Save prediction results to unique folder for future inspections
    save_name = f'experiment_{int(time.time())}'
    os.makedirs(save_name, exist_ok=True)

    np.save(f'{save_name}/predictions.npy', predictions)

    with open(f'{save_name}/llm_outputs.txt', 'w') as file:
        file.writelines('\n\n'.join(outputs))

    labels = dataset['label'].to_numpy().astype(int)

    print(classification_report(labels, predictions, target_names=['Exclusion', 'Inclusion']))
    print(f'MCC: {matthews_corrcoef(labels, predictions)}')


def main(backend: str, bit: Optional[int], dataset: str):
    assert backend in ['hf', 'unsloth'], 'Invalid backend specified.'
    assert bit in [4, 8, 16], 'Invalid quantization configuration.'

    dataset = build_dataset(dataset)

    if backend == 'hf':
        if bit == 4:
            q_config = BitsAndBytesConfig(load_in_4bit=True)
        elif bit == 8:
            q_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            q_config = BitsAndBytesConfig()

        pipe = pipeline(
            "text-generation",
            model="google/medgemma-27b-text-it",
            torch_dtype=torch.bfloat16,
            model_kwargs={"quantization_config": q_config}
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
            max_seq_length=32768,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma3",
        )

        pipe = (model, tokenizer)

    evaluate(dataset, pipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run evaluation experiment."
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        default="hf",
        help="Backend to use, currently support hf (huggingface default) or unsloth (faster but limited model options)"
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=None,
        help="Quantization to use (4, 8). If omitted, defaults to BF16."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='dataset.csv',
        required=True,
        help="Dataset path."
    )

    args = parser.parse_args()
    main(args.backend, args.bit, args.dataset)
