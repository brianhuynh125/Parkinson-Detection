import pandas as pd
from pathlib import Path

paths, labels = [], []
for cls in ['PD_AH', 'HC_AH']:
    for f in Path('rawdata', cls).glob('*.wav'):
        paths.append(f)
        labels.append(1 if cls == 'PD_AH' else 0)

df = pd.DataFrame({'path': paths, 'label': labels})
df.to_csv('dataset_manifest.csv', index=False)
print(df.head())