from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def parse_gemma_output(output: str) -> int:
    eol = output[:10]  # This number can be lower if we can guarantee the model would only say yes/no at the very end

    eol = eol.lower().strip()

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

    ehr_texts, labels = raw_dataset['text'], raw_dataset['labels']

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


def run(prompt: str, model, tokenizer):
    # Execute a single prompt and get parsed response from LLM

    messages = [
        {"role": "system", "content": "You are a clinical decision support assistant. Your task is to help determine whether HIV testing is recommended for a patient based on Dutch medical notes (EHR text). You follow clinical guidelines and reason step by step before making a decision. Only recommend testing if there is a clear indicator."},
        {"role": "user", "content": prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    )

    gen_tokens = outputs[0][inputs["input_ids"].size(-1):]

    response = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True,
    )

    decision = parse_gemma_output(response)

    return decision, response


def evaluate(dataset: str | pd.DataFrame, model, tokenizer):
    # Execute a prompt dataset and evaluate against the provided labels

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    predictions, outputs = [], []

    for prompt, _ in tqdm(dataset.iterrows()):
        prediction, output = run(prompt, model, tokenizer)

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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/medgemma-27b-text-it-GGUF",
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_8bit=True
    )

    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="google/medgemma-27b-text-it",
    )

    dataset = build_dataset('dataset.csv')

    evaluate(dataset, model, tokenizer)


if __name__ == '__main__':
    main()
