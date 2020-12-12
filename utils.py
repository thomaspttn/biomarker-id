import pandas as pd
from sklearn import preprocessing
import numpy as np

str_cols = [
    'Cancer Type Detailed',
    'Cellularity',
    'Chemotherapy',
    'ER Status',
    'ER status measured by IHC',
    'HER2 Status',
    'HER2 status measured by SNP6',
    'Hormone Therapy',
    'Inferred Menopausal State',
    'Integrative Cluster',
    'Oncotree Code',
    'Pam50 + Claudin-low subtype',
    'PR Status',
    'Primary Tumor Laterality',
    'Radio Therapy',
    'Tumor Other Histologic Subtype',
    'Type of Breast Surgery',
    'Patient\'s Vital Status'
]

# Reads in datasets (had to comment out current and uncomment next and rerun)
df = pd.read_csv('/biomarker-id/data/brca-data.csv')
#df = pd.read_csv('/biomarker-id/data/CDH1-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/GATA3-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/KMT2C-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/KMT2D-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/MAP3K1-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/NOTCH1-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/PDE4DIP-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/PIK3CA-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/TBX3-dataset.csv')
#df = pd.read_csv('/biomarker-id/data/TP53-dataset.csv')

# Drop rows which are missing data
df.replace('NA', np.nan)
df = df.dropna()

# Drop death from other causes
df = df[df['Patient\'s Vital Status'] != 'Died of Other Causes']
print(df.shape)

print(df)
# Map nominal features to numbers
for col in str_cols:
    df[col] = pd.Categorical(df[col], ordered=True).codes
df = df.apply(pd.to_numeric)

# Normalize all features to be between 0 and 1
np_df = df.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(np_df)

# Extract final df (had to comment out current and uncomment next and rerun)
norm_df = pd.DataFrame(scaled)
norm_df.to_csv('/biomarker-id/data/brca-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/CDH1-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data//GATA3-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/KMT2C-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/KMT2D-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/MAP3K1-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/NOTCH1-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/PDE4DIP-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/PIK3CA-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/TBX3-normalized.csv', index=False)
#norm_df.to_csv('/biomarker-id/data/TP53-normalized.csv', index=False)
print(norm_df)