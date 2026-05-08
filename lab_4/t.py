import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score


df = pd.read_csv('Depression Student Dataset.csv')
df = pd.get_dummies(df, columns=['Gender', 'Dietary Habits','Have you ever had suicidal thoughts ?','Family History of Mental Illness'], drop_first=True)
X = df.drop(['Depression','Sleep Duration'], axis=1)
y = (df['Depression'] == 'Yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print("=== Random Forest ===")
print("OOB Accuracy:", rf.oob_score_)
print("OOB Error:", 1 - rf.oob_score_)

y_pred_rf = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_rf))

ada = AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("\n=== AdaBoost ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred_ada))

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("\n=== Gradient Boosting ===")
print("Test Accuracy:", accuracy_score(y_test, y_pred_gb))

plt.figure(figsize=(10, 7))

def plot_roc(model, name):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')

plot_roc(rf, 'RF')
plot_roc(ada, 'Ada')
plot_roc(gb, 'GB')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Ложные срабатывания (FPR)')
plt.ylabel('Верные срабатывания (TPR)')
plt.title('ROC-кривые')
plt.legend(loc='lower right')
plt.grid(alpha=0.2)
plt.show()
