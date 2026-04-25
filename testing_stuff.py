import pandas as pd
df = pd.read_csv("mlb_fa_2026.csv")
print(df["Position"].value_counts())


train = pd.read_csv("mlb_fa_training_data_v1.csv")
print(train["Position"].value_counts())