from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd

# take 4 features
trainData = pd.read_csv("./msbd5001-fall2019/train.csv", parse_dates=["purchase_date", "release_date"])
trainData = trainData.dropna(axis=0, how='any')
trainData['buyDays'] = trainData['purchase_date'] - trainData['release_date']
trainData['buyDays'] = trainData['buyDays'].dt.days
X_train = trainData[['total_positive_reviews', 'total_negative_reviews', 'price', 'buyDays']]
y_train = trainData['playtime_forever']

print('X_train', X_train)
print('Y_train', y_train)

# Fit tree regression model
model1 = DecisionTreeRegressor(max_depth=4)
model1 = model1.fit(X_train, y_train)
testData = pd.read_csv("./msbd5001-fall2019/test.csv", parse_dates=["purchase_date", "release_date"])

# fill in the NaN position
for col in list(testData.columns[testData.isnull().sum() > 0]):
    mean_val = testData[col].mean()
    testData[col].fillna(mean_val, inplace=True)

# do the prediction
testData['buyDays'] = testData['purchase_date'] - testData['release_date']
testData['buyDays'] = testData['buyDays'].dt.days
X_test = testData[['total_positive_reviews', 'total_negative_reviews', 'price', 'buyDays']]
y_test = model1.predict(X_test)

print('X_test', X_test)
print('y_test', y_test)

id = testData.loc[:,'id']
df = pd.DataFrame({'id':id, 'playtime_forever':y_test})
df.to_csv('result2.csv', index=False)

with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(model1, out_file = f)


