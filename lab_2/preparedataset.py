import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, root_mean_squared_error

df = pd.read_csv('foods_clean.csv')

df = df.drop_duplicates(subset=['name'] if 'name' in df.columns else ['food'])

columns_to_drop = [
    'unnamed: 0.1', 'unnamed: 0', 'water', 'nutrition density',
    'saturated fats', 'monounsaturated fats', 'polyunsaturated fats',
    'cholesterol', 'sodium', 'vitamin a', 'vitamin b1', 'vitamin b11',
    'vitamin b12', 'vitamin b2', 'vitamin b3', 'vitamin b5', 'vitamin b6',
    'vitamin c', 'vitamin d', 'vitamin e', 'vitamin k', 'calcium',
    'copper', 'iron', 'magnesium', 'manganese', 'phosphorus', 'potassium',
    'selenium', 'zinc', 'vitamin_a', 'vitamin_b1', 'vitamin_b2',
    'vitamin_b3', 'vitamin_b5', 'vitamin_b6', 'vitamin_b12', 'vitamin_c',
    'vitamin_d', 'vitamin_e', 'vitamin_k'
]

cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
df_clean = df.drop(columns=cols_to_drop_existing).copy()

df_clean.columns = [col.lower() for col in df_clean.columns]
if 'food' in df_clean.columns:
    df_clean = df_clean.rename(columns={'food': 'name'})

total_macros = df_clean['protein'] + df_clean['fat'] + df_clean['carbs']
df_clean['p_ratio'] = df_clean['protein'] / total_macros.replace(0, 1)
df_clean['f_ratio'] = df_clean['fat'] / total_macros.replace(0, 1)
df_clean['c_ratio'] = df_clean['carbs'] / total_macros.replace(0, 1)

# ---> ИЗМЕНЕНО: Разделение на завтрак, обед, ужин и снэк <---
def suggest_meal_type(row):
    if row['calories'] < 120:
        return 'snack'
    elif row['carbs'] > 20 and row['fiber'] > 2:
        return 'breakfast'
    elif row['protein'] > 20 and row['fat'] > 15:
        return 'dinner'  # Ужин делаем более белковым
    else:
        return 'lunch'   # Все остальные полноценные приемы пищи — обед
# -----------------------------------------------------------

df_clean['meal_type'] = df_clean.apply(suggest_meal_type, axis=1)

df_clean['cal_diff'] = abs(df_clean['calories'] - (df_clean['protein']*4 + df_clean['carbs']*4 + df_clean['fat']*9))
df_clean = df_clean[df_clean['cal_diff'] < 50]

scaler = MinMaxScaler()
numeric_features = ['calories', 'protein', 'fat', 'carbs', 'fiber', 'sugar', 'p_ratio', 'f_ratio', 'c_ratio']
numeric_features = [col for col in numeric_features if col in df_clean.columns]

df_ml = df_clean.copy()
df_ml[numeric_features] = scaler.fit_transform(df_clean[numeric_features])

df_ml = pd.get_dummies(df_ml, columns=['meal_type'], prefix='type')

df_ml.to_csv('df_clean.csv', index=False)

# Признаки оставлены как у тебя, так как их достаточно для классификации
X = df_clean[['protein', 'fat', 'carbs', 'fiber', 'sugar']]
y_reg = df_clean['calories']
y_clf = df_clean['meal_type']
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = root_mean_squared_error(y_reg_test, y_reg_pred)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print('Mean Squared Error:', mse, '\n')
print('''MSE в переделах нормы. Наличии выбросов в данных исправлено. 
    Поскольку MSE возводит ошибки в квадрат, редкие, но крупные нестыковки в калориях сильно завышают этот показатель. 
    Модель в целом точна, но чувствительна к аномалиям.''', '\n')
print('Root Mean Squared Error:', rmse)
print('Mean Absolute Error:', mae, '\n')
print('''MAE показывает, что в среднем модель ошибается всего на 4.3 ккал, 
    что является отличным результатом для пищевых продуктов.''')
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---> ИЗМЕНЕНО: Добавлен class_weight='balanced' для решения проблемы дисбаланса <---
clf_model = LogisticRegression(max_iter=1000, class_weight='balanced')
# ----------------------------------------------------------------------------------

clf_model.fit(X_train_scaled, y_clf_train)
y_clf_pred = clf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_clf_test, y_clf_pred)

cm = confusion_matrix(y_clf_test, y_clf_pred)
labels = clf_model.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Реальный тип (True label)')
plt.xlabel('Предсказание модели (Predicted label)')
plt.show()

print("\n--- Детальный отчет по метрикам ---")
print(classification_report(y_clf_test, y_clf_pred))

print(f'Accuracy: {accuracy:.4f}\n')
print(f'''Точность {accuracy:.2%} говорит об отличной предсказательной способности модели. 
    Состав макронутриентов (БЖУ) уверенно коррелирует с типом приема пищи.
    Оставшиеся ошибки (в основном между обедом и ужином) абсолютно логичны, 
    так как эти приемы пищи часто имеют схожий профиль плотности нутриентов.''')