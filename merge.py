import pandas as pd
import re


icd_file = pd.read_csv('data/icd_c.csv')
icd_file['HIV_indicator_HIVteam'] = icd_file['HIV_indicator_HIVteam'].astype(int)

files = [f'data/Datauitgifte_AwareHIV_deidentified_chunk{x}.csv' for x in range(1, 4)]
dfs = [pd.read_csv(f, index_col=0) for f in files]

df = pd.concat(dfs)

df['authored'] = pd.to_datetime(df['authored'])

df['ln'] = range(len(df))

df_sorted = df.sort_values(by=['Pseudoniem', 'authored', 'ln'], ascending=[True, False, True])

df_sorted['section_text'] = df_sorted['section_text'].apply(lambda x: x if str(x).endswith('\n') else str(x) + '\n')

result_df = df_sorted.groupby('Pseudoniem')['section_text'].apply(''.join).reset_index()

# Count how many section_text entries were merged per ID
counts = df_sorted.groupby('Pseudoniem').size().reset_index(name='num_entries')
result_df = result_df.merge(counts, on='Pseudoniem', how='left')

result_df['section_text'] = result_df['section_text'].apply(lambda x: re.sub(r'\n+', '\n', x))

result_df['section_text'] = result_df['section_text'].str.replace(r'\\n|\\t', ' ', regex=True)
result_df['section_text'] = result_df['section_text'].str.replace(r'\s+', ' ', regex=True)
result_df['section_text'] = result_df['section_text'].str.strip()
merged_df = result_df.merge(icd_file, on='Pseudoniem', how='inner')

merged_df.rename(columns={'HIV_indicator_HIVteam': 'flag', 'section_text': 'text'}, inplace=True)

merged_df.to_csv('dataset.csv')
