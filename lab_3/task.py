import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('indian_roads_dataset.csv')

print("Пропуски до обработки:")
print(df.isnull().sum())
print(df.columns.tolist())
print()
print(df.head())

df = pd.get_dummies(df, columns=['road_type','weather','visibility','traffic_density','cause','accident_severity'], drop_first=True)

X = df.drop(['city', 'state','latitude','longitude','date','time','day_of_week','festival','risk_score'], axis=1)
y = df['risk_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

reg_model = DecisionTreeRegressor(random_state = 42, max_depth = 5)
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE:', mse)
print('R2:', r2)

plt.figure(figsize=(20, 10))
plot_tree(reg_model, feature_names=X.columns, class_names=['Low', 'High'],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Regressor")
plt.show()