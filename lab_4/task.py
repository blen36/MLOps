import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

df = pd.read_csv("dataset.csv")
df = df.dropna()
df = pd.get_dummies(df)

# 3. РАЗДЕЛЕНИЕ (укажи имя целевой колонки вместо 'target')
target_column = 'target'
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
# Random Forest (Бэггинг) - уменьшает дисперсию
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

# AdaBoost (Бустинг на весах) - исправляет ошибки
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# Gradient Boosting (Бустинг на остатках/антиградиенте)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# 5. ОЦЕНКА
print(f"RF OOB Score: {rf.oob_score_:.3f}") # Оценка на 36.8% данных

for name, model in [("RF", rf), ("Ada", ada), ("GB", gb)]:
    probs = model.predict_proba(X_test)[:, 1]
    print(f"{name} | Test AUC: {roc_auc_score(y_test, probs):.3f}")