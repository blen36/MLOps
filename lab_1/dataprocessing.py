import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("melb_data.csv")
print("Первые 5 строк:")
print(df.head())
print()
print("Информация о датасете:")
print(df.info())
print()

#ЗАПОЛНЕНИЕ ПРОПУСКОВ
print("Количество пропусков по столбцам:")
for col in df:
    nan_value = df[col].isnull().sum()
    print(f"column {col} has {nan_value} nan-values {df[col].dtype} ")

df[['Car', 'BuildingArea', 'YearBuilt']] = df[['Car', 'BuildingArea', 'YearBuilt']].fillna(df[['Car', 'BuildingArea', 'YearBuilt']].median())
df['CouncilArea']=df['CouncilArea'].fillna(df['CouncilArea'].mode()[0])

print("Пропуски после заполнения:")
print(df[['Car','BuildingArea','YearBuilt','CouncilArea']].isnull().sum())
print()

#НОРМАЛИЗАЦИЯ ДАННЫХ
df = df.drop(['Address', 'Date'], axis=1)
print("Размер датасета до OHE:", df.shape)
cat_cols = df.select_dtypes(include=['object', 'str']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("Размер датасета после OHE:", df.shape)
print()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

train_df.to_csv("train_processed.csv", index=False)
test_df.to_csv("test_processed.csv", index=False)

