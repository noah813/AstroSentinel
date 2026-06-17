import pandas as pd
import os

d = 'C:/Users/noahl/documents/AstroSentinel/data/chunks'
for i in range(366, 417):
    f_in = f"{d}/chunk_{i}.csv"
    f_out = f"{d}/result_{i}.csv"
    if os.path.exists(f_in):
        df = pd.read_csv(f_in)
        labels = []
        for c, z in zip(df['concentration'], df['zero_like_ratio']):
            if c > 2.0 or z > 0.5:
                labels.append('bot')
            elif c <= 1.5 and z < 0.2:
                labels.append('normal')
            else:
                labels.append('pending')
        out_df = pd.DataFrame({'author_id': df['author_id'], 'label': labels})
        out_df.to_csv(f_out, index=False)
