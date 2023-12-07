import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('SeoulHourlyAvgAirPollution.csv')

print(df.head())
print(df.info())

df = df.dropna()

X = df[['이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스(ppm)', '미세먼지(㎍/㎥)']]
y = df['초미세먼지(㎍/㎥)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

average_accuracy = results.mean()
print(f'평균 정확도 (k-fold): {average_accuracy}')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
