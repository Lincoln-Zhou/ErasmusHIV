import pandas as pd
import json


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


def calculate_cumulate_logprob(response):
    # Calculates the cumulated log-probability of a generation, normalized by output token length
    # Can be used as a metric for test time scaling
    # Experimental
    probs = response['choices'][0]['logprobs']['content']

    cumulate_logprob = sum(x['logprob'] for x in probs if x['logprob'] is not None) / len(probs)

    return cumulate_logprob


def build_dataset(raw_dataset: str | pd.DataFrame) -> pd.DataFrame:
    # Convert raw dataset into prompt based

    if isinstance(raw_dataset, str):
        raw_dataset = pd.read_csv(raw_dataset)

    ehr_texts, labels = raw_dataset['text'], raw_dataset['flag']

    ehr_prompts = ehr_texts.apply(lambda x:
                                  f"""
    Text:
    "{x}"
    """)

    prompt_dataset = pd.DataFrame(data={'prompt': ehr_prompts, 'label': labels})

    return prompt_dataset
