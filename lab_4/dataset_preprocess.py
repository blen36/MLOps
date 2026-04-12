import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bank.csv")

print("Первые строки:")
print(df.head(), "\n")

print("Размер датасета:", df.shape, "\n")

print("Типы данных:")
print(df.info(), "\n")

print("Статистика:")
print(df.describe(), "\n")

print("Пропуски:")
print(df.isnull().sum(), "\n")

print("Распределение target:")
print(df['deposit'].value_counts())

print("Дубликаты:", df.duplicated().sum())
df = df.drop_duplicates()

df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.show()

df.to_csv("bank_preprocessed.csv", index=False)