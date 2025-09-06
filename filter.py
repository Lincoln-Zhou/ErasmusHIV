import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


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

df.to_csv('dataset.csv', index=False)

train, test = train_test_split(df, test_size=0.1, stratify=df['flag'], random_state=24)

train.to_csv('train.csv')
test.to_csv('test.csv')
