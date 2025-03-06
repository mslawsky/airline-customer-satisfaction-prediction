#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a random forest model (Fixed Version)

# ## **Introduction**
# 
# As you're learning, random forests are popular statistical learning algorithms. Some of their primary benefits include reducing variance, bias, and the chance of overfitting.
# 
# This activity is a continuation of the project you began modeling with decision trees for an airline. Here, you will train, tune, and evaluate a random forest model using data from spreadsheet of survey responses from 129,880 customers. It includes data points such as class, flight distance, and inflight entertainment. Your random forest model will be used to predict whether a customer will be satisfied with their flight experience.
# 
# **Note:** This version of the notebook addresses a critical data leakage issue that affected the original implementation.

# ## **Step 1: Imports** 

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
air_data = pd.read_csv("Invistico_Airline.csv")

# ## **Step 2: Data cleaning** 

# Display first 10 rows
print("First 10 rows of the dataset:")
print(air_data.head(10))

# Display variable names and types
print("\nVariable names and data types:")
print(air_data.dtypes)

# Check dataset size
print("\nDataset shape:")
print(f"Number of rows: {air_data.shape[0]}")
print(f"Number of columns: {air_data.shape[1]}")

# Check for missing values and drop them
missing_rows = air_data.isna().any(axis=1).sum()
print(f"\nNumber of rows with missing values: {missing_rows}")

air_data_subset = air_data.dropna()
print(f"Shape after dropping missing values: {air_data_subset.shape}")

# Display the cleaned data
print("\nFirst 10 rows after cleaning:")
print(air_data_subset.head(10))

# Convert categorical features to one-hot encoded features
air_data_subset_dummies = pd.get_dummies(air_data_subset)

print("\nVariable names after one-hot encoding:")
print(air_data_subset_dummies.columns.tolist())

# ## **DATA LEAKAGE FIX**
# 
# # In the previous implementation, we had a critical issue: the one-hot encoding created two columns 
# # from the satisfaction variable: 'satisfaction_satisfied' (our target) and 'satisfaction_dissatisfied'.
# # Since these columns contain exactly the same information (just inverted), keeping 'satisfaction_dissatisfied'
# # in our feature set leads to data leakage - the model can perfectly predict the target using this column.
# 
# # We fix this by:
# # 1. First extracting our target variable
y = air_data_subset_dummies['satisfaction_satisfied']
# # 2. Then removing BOTH satisfaction columns from features to prevent leakage
X = air_data_subset_dummies.drop(['satisfaction_satisfied', 'satisfaction_dissatisfied'], axis=1)

print("\nAfter fixing data leakage:")
print(f"Number of features: {X.shape[1]}")
print("Checking if satisfaction columns remain in features:")
print(f"'satisfaction_satisfied' in features: {'satisfaction_satisfied' in X.columns}")
print(f"'satisfaction_dissatisfied' in features: {'satisfaction_dissatisfied' in X.columns}")

# ## **Step 3: Model building** 

# Split the data into train, validate, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print(f"\nTraining set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Set hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create list of split indices for validation
X_train_val_combined = pd.concat([X_train, X_val])
y_train_val_combined = pd.concat([y_train, y_val])

# Create test_fold array: -1 for training set, 0 for validation set
test_fold = np.zeros(X_train_val_combined.shape[0])
test_fold[:X_train.shape[0]] = -1
ps = PredefinedSplit(test_fold)

# Instantiate and train the model
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV to search over specified parameters
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=ps,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit the model
print("\nTraining the model with GridSearchCV (this may take some time)...")
grid_search.fit(X_train_val_combined, y_train_val_combined)

# Get optimal parameters
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# ## **Step 4: Results and evaluation** 

# Use the optimal parameters
best_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)

# Fit the optimal model
best_rf.fit(X_train_val_combined, y_train_val_combined)

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Get performance scores
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dissatisfied', 'Satisfied'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the figure
plt.show()

# Display feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
plt.barh(range(10), top_features['Importance'], align='center')
plt.yticks(range(10), top_features['Feature'])
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.savefig('feature_importance.png')  # Save the figure
plt.show()

print("\nTop 10 Most Important Features:")
print(top_features)

# Create table of results
results = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'Accuracy', 'F1 Score'],
    'Score': [precision, recall, accuracy, f1]
})

print("\nResults Table:")
print(results)

# Save the model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

print("\nModel saved as 'random_forest_model.pkl'")

print("""
## Data Leakage Explanation

In our initial implementation, we encountered a data leakage issue that led to artificially perfect model performance 
(precision, recall, accuracy, and F1 score all 1.0000). This occurred because:

1. When we one-hot encoded the categorical variables, our target variable 'satisfaction' was converted into two binary columns:
   - 'satisfaction_satisfied' (our target)
   - 'satisfaction_dissatisfied' (which contained exactly the same information as the target, just inverted)

2. By including 'satisfaction_dissatisfied' in our feature set, we inadvertently gave the model access to the target 
   variable during training and testing. This is a classic case of data leakage where the model learns to use information 
   that wouldn't be available in a real prediction scenario.

3. In this fixed implementation, we explicitly removed both satisfaction columns from the feature set, then used only 
   'satisfaction_satisfied' as our target variable.

Benefits of fixing this data leakage:

1. More realistic and honest evaluation of model performance
2. Better understanding of which features truly drive customer satisfaction
3. A model that can genuinely generalize to new, unseen data
4. Meaningful feature importance rankings that can guide business decisions

The model now shows more realistic performance metrics (likely in the 0.85-0.95 range instead of perfect 1.0000 scores), 
which better represents its true predictive capability in real-world applications.
""")
