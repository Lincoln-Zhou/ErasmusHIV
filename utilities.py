import pandas as pd
from textwrap import dedent
import json
import smtplib
from email.message import EmailMessage
from config import MAIL_FROM, MAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD


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


def format_group(df: pd.DataFrame, preserved_cols, rename: dict = None) -> str:
    # Turn a grouped DataFrame into a multiline formatted string.

    lines = []

    for row in df.itertuples(index=False):
        if rename is None:
            pairs = [f"{col}: {getattr(row, col)}" for col in preserved_cols]
        else:
            pairs = [f"{rename[col]}: {getattr(row, col)}" for col in preserved_cols]

        lines.append(", ".join(pairs))

    return "\n".join(lines)


def build_dataset_with_add(raw_dataset: str | pd.DataFrame, med_data: str | pd.DataFrame, test_data: str | pd.DataFrame) -> pd.DataFrame:
    # Inclusion of medication and lab test data are experimental. Currently, they are only supported by passing in pandas DataFrame

    if isinstance(raw_dataset, str):
        raw_dataset = pd.read_csv(raw_dataset)

    if isinstance(med_data, str):
        med_data = pd.read_csv(med_data)

    if isinstance(test_data, str):
        test_data = pd.read_csv(test_data)

    med_fmt = (
        med_data
        .groupby('Pseudoniem', as_index=False)
        .apply(lambda g: format_group(g, preserved_cols=['code5_ATC_code', 'code_text'], rename={'code5_ATC_code': 'ATC', 'code_text': 'Medication'}))
        .rename(columns={None: 'med_str'})
    )

    test_fmt = (
        test_data
        .groupby('Pseudoniem', as_index=False)
        .apply(lambda g: format_group(g, preserved_cols=['hix_code', 'valueString'], rename={'hix_code': 'Test', 'valueString': 'Result'}))
        .rename(columns={None: 'test_str'})
    )

    merged = (
        raw_dataset
        .merge(med_fmt, on='Pseudoniem', how='left')
        .merge(test_fmt, on='Pseudoniem', how='left')
    )

    def build_template(row):
        data_block = row['med_str'] if pd.notna(row['med_str']) else "None"
        test_block = row['test_str'] if pd.notna(row['test_str']) else "None"

        return dedent(f"""\
        EHR text:
        {row['text']}
        
        medication data:
        {data_block}
        
        lab test data:
        {test_block}
        """)    # We should inspect whether calling dedent() on a f-string would cause formatting issues

    merged['combined'] = merged.apply(build_template, axis=1)

    labels = raw_dataset['flag']

    prompt_dataset = pd.DataFrame(data={'prompt': merged['combined'], 'label': labels})

    return prompt_dataset


def build_dataset(raw_dataset: str | pd.DataFrame) -> pd.DataFrame:
    # Convert raw dataset into prompt based

    if isinstance(raw_dataset, str):
        raw_dataset = pd.read_csv(raw_dataset)

    ehr_texts, labels = raw_dataset['text'], raw_dataset['flag']

    ehr_prompts = ehr_texts.apply(lambda x:
                                  dedent(f"""\
    Text:
    "{x}"
    """))

    prompt_dataset = pd.DataFrame(data={'prompt': ehr_prompts, 'label': labels})

    return prompt_dataset


def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
