import pandas as pd
import re


ph_codes = [
    "A22.1", "A31", "A31.0", "A31.1", "A31.8", "A31.9", "A37.1", "A39.8",
    "A43.0", "A48.1", "B25.0", "B97.2", "J10.0", "J11.0", "J12.0", "J12.1",
    "J12.2", "J12.3", "J12.8", "J12.9", "J13", "J14", "J15", "J15.0", "J15.1",
    "J15.2", "J15.3", "J15.4", "J15.5", "J15.6", "J15.7", "J15.8", "J15.9",
    "J16", "J16.0", "J16.8", "J17", "J17.0", "J17.1", "J17.2", "J17.3", "J17.8",
    "J18", "J18.0", "J18.1", "J18.8", "J18.9", "J22", "J84.9", "J85.1", "J86.0",
    "J86.9", "J96.0", "J96.9", "J98.8", "U07.1"
]

icd_file = pd.read_csv('data/icd10.csv')
icd_file['HIV_indicator_HIVteam'] = icd_file['HIV_indicator_HIVteam'].astype(int)

icd_file = icd_file[icd_file['HIV_indicator_HIVteam'].isin([0, 1])]
icd_file = icd_file[icd_file['icd10_code'].isin(ph_codes)]

icd_file = icd_file.groupby('Pseudoniem', as_index=False)['HIV_indicator_HIVteam'].min()

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

merged_df.rename(columns={'HIV_indicator_HIVteam': 'labels', 'section_text': 'text'}, inplace=True)

merged_df.to_csv('dataset.csv')
