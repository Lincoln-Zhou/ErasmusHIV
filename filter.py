import pandas as pd
import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

df = pd.read_csv('dataset.csv')

enc = tokenizer.batch_encode_plus(
    df["text"].astype(str).tolist(),
    add_special_tokens=True,
    truncation=False,
    return_attention_mask=False,
    return_token_type_ids=False
)

df['token_length'] = [len(ids) for ids in enc["input_ids"]]

mu = np.mean(df['token_length'])
std = np.std(df['token_length'])

df = df[(df['token_length'] > mu - 3 * std) & (df['token_length'] < mu + 3 * std)]

df.to_csv('dataset_filtered.csv', index=False)
