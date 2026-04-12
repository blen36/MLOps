import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

df = pd.read_csv("bank_preprocessed.csv")

X = df.drop('deposit', axis=1)
y = df['deposit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print("=== Random Forest ===")
print("OOB Accuracy:", rf.oob_score_)
print("OOB Error:", 1 - rf.oob_score_)

rf_pred = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, rf_pred))

ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
ada.fit(X_train, y_train)

ada_pred = ada.predict(X_test)
print("\n=== AdaBoost ===")
print("Test Accuracy:", accuracy_score(y_test, ada_pred))

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
print("\n=== Gradient Boosting ===")
print("Test Accuracy:", accuracy_score(y_test, gb_pred))

rf_probs = rf.predict_proba(X_test)[:, 1]
ada_probs = ada.predict_proba(X_test)[:, 1]
gb_probs = gb.predict_proba(X_test)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

rf_auc = auc(rf_fpr, rf_tpr)
ada_auc = auc(ada_fpr, ada_tpr)
gb_auc = auc(gb_fpr, gb_tpr)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot(ada_fpr, ada_tpr, label=f'AdaBoost (AUC = {ada_auc:.3f})')
plt.plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC = {gb_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые моделей')
plt.legend()
plt.grid()
plt.show()