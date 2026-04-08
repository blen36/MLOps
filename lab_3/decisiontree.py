import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

df = pd.read_csv('df_clean.csv')
X = df.drop(columns=['calories', 'name', 'cal_diff'])
y_reg = df['calories']
threshold = y_reg.quantile(0.75)
y_clf = (y_reg > threshold).astype(int)

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

reg_model = DecisionTreeRegressor(random_state=42, max_depth=5)
reg_model.fit(X_train, y_reg_train)
y_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_pred)
r2 = r2_score(y_reg_test, y_pred)
print("MSE:", mse)
print("R2:", r2)

clf_model = DecisionTreeClassifier( random_state=42,
    max_depth=3,
    min_samples_leaf=10,
    min_samples_split=20)

clf_model.fit(X_train, y_clf_train)
y_proba = clf_model.predict_proba(X_test)

y_clf_pred = clf_model.predict(X_test)
print("Матрица ошибок: (строки = реальные значения; столбцы = предсказания модели)")
cm = confusion_matrix(y_clf_test, y_clf_pred)
print(cm)

fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
