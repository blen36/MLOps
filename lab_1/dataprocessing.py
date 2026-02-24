import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
print(df.head())
print()
print(df.info())
print()
cols = df.columns.tolist()
print(cols)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

nan_matrix = df.isnull()
print(nan_matrix)

for col in df:
    nan_value = df[col].isnull().sum()
    print(f"column {col} has {nan_value} values")

age_median = df['Age'].median()
cabin_mode = df['Cabin'].mode()[0]
embarked_mode = df['Embarked'].mode()[0]

df['Cabin'].fillna(cabin_mode, inplace=True)
df['Age'].fillna(age_median, inplace=True)
df['Embarked'].fillna(embarked_mode, inplace=True)

for col in df:
    nan_value = df[col].isnull().sum()
    print(f"column {col} has {nan_value} values")