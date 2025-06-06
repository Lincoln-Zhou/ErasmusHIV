from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
import numpy as np
from tqdm import tqdm


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


def run(prompt: str):
    # Execute a single prompt and get parsed response from LLM
    pass


def evaluate(dataset: str | pd.DataFrame):
    # Execute a prompt dataset and evaluate against the provided labels

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    predictions, outputs = [], []

    for prompt, _ in tqdm(dataset.iterrows()):
        prediction, output = run(prompt)

        predictions.append(prediction)
        outputs.append(output)

    predictions = np.array(predictions).astype(int)

    np.save('predictions.npy', predictions)

    with open('llm_outputs.txt', 'w') as file:
        file.writelines('\n\n'.join(outputs))

    labels = dataset['label'].to_numpy().astype(int)

    print(classification_report(labels, predictions, target_names=['Exclusion', 'Inclusion']))
    print(f'MCC: {matthews_corrcoef(labels, predictions)}')
