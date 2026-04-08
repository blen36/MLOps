import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Employee.csv')

print("Пропуски до обработки:")
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['EverBenched', 'City', 'Gender', 'Education', 'JoiningYear'], drop_first=True)
X = df.drop(['LeaveOrNot'], axis=1)
y = df['LeaveOrNot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Сравнение разных значений C ===")

for C in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"C={C}: accuracy={acc:.4f}")


clf_model = LogisticRegression(C=0.01, max_iter=1000, random_state=42, class_weight='balanced')

clf_model.fit(X_train_scaled, y_train)
y_clf_pred = clf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_clf_pred)
cm = confusion_matrix(y_test, y_clf_pred)
print(f"\nНаилучшая точность модели с разными C: {accuracy:.4f}")
print("\nМатрица ошибок:")
print(cm)

labels = ['Не уволился (0)', 'Уволился (1)']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Матрица ошибок (Confusion Matrix)')
plt.ylabel('Реальный тип (True label)')
plt.xlabel('Предсказание модели (Predicted label)')
plt.show()

print("\nОтчет классификации:")
print(classification_report(y_test, y_clf_pred))
