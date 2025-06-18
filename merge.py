import pandas as pd
import re
from sklearn.model_selection import train_test_split


icd_file = pd.read_csv('data/icd10.csv')
icd_file['HIV_indicator_HIVteam'] = icd_file['HIV_indicator_HIVteam'].astype(int)

icd_file = icd_file[icd_file['HIV_indicator_HIVteam'].isin([0, 1])]

icd_file = icd_file.groupby('Pseudoniem', as_index=False)['HIV_indicator_HIVteam'].max()

files = [f'data/Datauitgifte_AwareHIV_deidentified_chunk{x}.csv' for x in range(1, 4)]
dfs = [pd.read_csv(f, index_col=0) for f in files]

df = pd.concat(dfs)

df['authored'] = pd.to_datetime(df['authored'])

df['ln'] = range(len(df))

df_sorted = df.sort_values(by=['Pseudoniem', 'authored', 'ln'], ascending=[True, False, True])

df_sorted['section_text'] = df_sorted['section_text'].apply(lambda x: x if str(x).endswith('\n') else str(x) + '\n')

result_df = df_sorted.groupby('Pseudoniem')['section_text'].apply(''.join).reset_index()

result_df['section_text'] = result_df['section_text'].apply(lambda x: re.sub(r'\n+', '\n', x))

result_df['section_text'] = result_df['section_text'].str.replace(r'\\n|\\t', ' ', regex=True)
result_df['section_text'] = result_df['section_text'].str.replace(r'\s+', ' ', regex=True)
result_df['section_text'] = result_df['section_text'].str.strip()

pretrain_merged_text = ' '.join(result_df['section_text'])

with open('pretrain_merged_text.txt', 'w') as file:
    file.write(pretrain_merged_text)

pretrained_by_patient = result_df[['Pseudoniem', 'section_text']]
pretrained_by_patient.to_csv('pretrained_by_patient.csv')

merged_df = result_df.merge(icd_file, on='Pseudoniem', how='inner')

merged_df.rename(columns={'HIV_indicator_HIVteam': 'flag', 'section_text': 'text'}, inplace=True)

merged_df.to_csv('dataset.csv')

train, test = train_test_split(merged_df, test_size=0.1, stratify=merged_df['flag'], random_state=24)

train.to_csv('train_pi.csv')
test.to_csv('test_pi.csv')
