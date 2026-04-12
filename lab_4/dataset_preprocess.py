import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")
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
print(df['target'].value_counts())

print("Дубликаты:", df.duplicated().sum())
df = df.drop_duplicates()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

categorical_cols = [
    'sex',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'ca',
    'thal'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Корреляционная матрица")
plt.show()

df.to_csv("heart_preprocessed.csv", index=False)
