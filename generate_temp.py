import pandas as pd
import re

files = [f'Datauitgifte_AwareHIV_deidentified_chunk{x}.csv' for x in range(1, 4)]
dfs = [pd.read_csv(f, index_col=0) for f in files]

df = pd.concat(dfs)

df['authored'] = pd.to_datetime(df['authored'])

df_sorted = df.sort_values(by=['Pseudoniem', 'authored', df.index.name], ascending=[True, False, True])

df_sorted['section_text'] = df_sorted['section_text'].apply(lambda x: x if str(x).endswith('\n') else str(x) + '\n')

result_df = df_sorted.groupby('Pseudoniem')['section_text'].apply(''.join).reset_index()

result_df['section_text'] = result_df['section_text'].apply(lambda x: re.sub(r'\n+', '\n', x))

result_df['PJP'] = ''
result_df['CAP'] = ''

# result_df[['Pseudoniem', 'section_text', 'PJP', 'CAP']].to_csv("annotation_template.csv", index=False)

with pd.ExcelWriter('annotation_template.xlsx', engine="xlsxwriter") as writer:
    writer.book.formats[0].set_text_wrap()  # Enable multi-line cell display
    result_df[['Pseudoniem', 'section_text', 'PJP', 'CAP']].to_excel(writer)
