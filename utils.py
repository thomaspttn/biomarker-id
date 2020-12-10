import pandas as pd
from sklearn import preprocessing

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
    'Overall Survival Status'
]

df = pd.read_csv('./data/brca-data.csv')
print(df.shape)

# Drop rows which are missing data
df = df.dropna()

# Map nominal features to numbers
for col in str_cols:
    df[col] = pd.Categorical(df[col], ordered=True).codes
df = df.apply(pd.to_numeric)

# Normalize all features to be between 0 and 1
np_df = df.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(np_df)

# Extract final df
norm_df = pd.DataFrame(scaled)
norm_df.to_csv('./data/brca-normalized.csv')
print(norm_df)
