import pandas as pd

df = pd.read_parquet("data/artifacts/options_clean.parquet")
print(df.head())
print(df.columns)
print(len(df))
