import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


main_file_path = 'data/train.csv'
data = pd.read_csv(main_file_path)

predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = data['SalePrice']
X = data[predictors]

# Fit a model using all the data
model_all_data = DecisionTreeRegressor()
model_all_data.fit(X, y)

print('Making predictions for the following five houses:')
print(X.head())
print(f'The previsions are: {model_all_data.predict(X.head())}')
print(f'The actual targets are:\n{data["SalePrice"].head()}')

# Fit a model using split data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
print(f'MAE: {mean_absolute_error(y_val, val_predictions)}')

# Loop through some different max_leaf_nodes to see what works best.
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return mae

results_info = []
for max_leaf_nodes in range(2, 500, 5):
    my_mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
    results_info.append((max_leaf_nodes, my_mae))
    print(f'Max leaf notes: {max_leaf_nodes}\t\tMean Absolute Errors: {my_mae}')

# Plot the max_leaf_nodes performance
fig, ax = plt.subplots()
ax.plot([item[0] for item in results_info], [item[1] for item in results_info],
        label='Val Data MAE')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Mean Absolute Error')
plt.show()

# See how a random forest model does

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
preds_val = forest_model.predict(X_val)
print(f'Random forest MAE: {mean_absolute_error(y_val, preds_val)}')

##########
# Handling missing data
##########
# Function to generate score based on input data
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)

# Option 1: Drop columns with missing values
# Set up data
data_targets = data['SalePrice']
data_predictors = data.drop(['SalePrice'], axis=1)
data_numeric_predictors = data.select_dtypes(exclude=['object'])
X_train, X_test, y_train, y_test = train_test_split(data_numeric_predictors, data_targets,
                                                    train_size=0.7, test_size=0.3,
                                                    random_state=0)

# Pull out rows with missing data
cols_with_missing_data = [col for col in X_train.columns if X_train[col].isnull().any()]
X_train_reduced = X_train.drop(cols_with_missing_data, axis=1)
X_test_reduced = X_test.drop(cols_with_missing_data, axis=1)

# Run the test
score = score_dataset(X_train_reduced, X_test_reduced, y_train, y_test)
print(f'Mean absolute error from dropping columns with missing values: {score}')







