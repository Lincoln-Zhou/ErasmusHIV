import pandas as pd


def parse_gemma_output(output: str) -> int:
    eol = output.strip()[-10:]  # This number can be lower if we can guarantee the model would only say yes/no at the very end

    eol = eol.lower()

    if 'yes' in eol:
        return 1
    elif 'no' in eol:
        return 0
    else:
        print(output)
        raise ValueError(f'Unrecognized output: {eol}')


def build_dataset(raw_dataset: str | pd.DataFrame) -> pd.DataFrame:
    # Convert raw dataset into prompt based

    if isinstance(raw_dataset, str):
        raw_dataset = pd.read_csv(raw_dataset)

    ehr_texts, labels = raw_dataset['text'], raw_dataset['flag']

    ehr_prompts = ehr_texts.apply(lambda x:
                                  f"""
    Analyze the following Dutch clinical note and determine whether HIV testing is recommended.

    Follow these steps:
    1. Identify any indicator condition(s) described.
    2. Check for valid exclusions.
    3. Decide whether HIV testing is warranted. Output only "YES" or "NO".

    Text:  
    "{x}"
    """)

    prompt_dataset = pd.DataFrame(data={'prompt': ehr_prompts, 'label': labels})

    return prompt_dataset
