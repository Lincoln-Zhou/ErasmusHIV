from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import pipeline, BitsAndBytesConfig


def parse_gemma_output(output: str) -> int:
    eol = output.strip()[-10:]  # This number can be lower if we can guarantee the model would only say yes/no at the very end

    eol = eol.lower()

    if 'yes' in eol:
        return 1
    elif 'no' in eol:
        return 0
    else:
        raise ValueError(f'Unrecognized output: {eol}')


def build_dataset(raw_dataset: str | pd.DataFrame) -> pd.DataFrame:
    # Convert raw dataset into prompt based

    if isinstance(raw_dataset, str):
        raw_dataset = pd.read_csv(raw_dataset)

    ehr_texts, labels = raw_dataset['text'], raw_dataset['flag']

    ehr_prompts = ehr_texts.apply(lambda x:
                                  f"""
    Analyze the following Dutch EHR text and determine whether HIV testing is recommended.  
    Follow these steps:  
    1. Analyze relevant clinical information.  
    2. Identify any applicable indicators.  
    3. Decide whether HIV testing is warranted. Output only "YES" or "NO".
    
    Text:  
    "{x}"
    """)

    prompt_dataset = pd.DataFrame(data={'prompt': ehr_prompts, 'label': labels})

    return prompt_dataset


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
        prediction, output = run(prompt, pipe)

        predictions.append(prediction)
        outputs.append(output)

    predictions = np.array(predictions).astype(int)

    np.save('predictions.npy', predictions)

    with open('llm_outputs.txt', 'w') as file:
        file.writelines('\n\n'.join(outputs))

    labels = dataset['label'].to_numpy().astype(int)

    print(classification_report(labels, predictions, target_names=['Exclusion', 'Inclusion']))
    print(f'MCC: {matthews_corrcoef(labels, predictions)}')


def main():
    pipe = pipeline(
        "text-generation",
        model="google/medgemma-27b-text-it",
        torch_dtype=torch.bfloat16,
        model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    )

    dataset = build_dataset('dataset.csv')

    evaluate(dataset, pipe)


if __name__ == '__main__':
    main()
