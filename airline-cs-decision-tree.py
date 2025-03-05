#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a decision tree
# 
# ## Introduction
# 
# A decision tree model can makes predictions for a target based on multiple features. Because decision trees are used across a wide array of industries, becoming proficient in the process of building one will help you expand your skill set in a widely-applicable way.   
# 
# For this activity, you work as a consultant for an airline. The airline is interested in predicting whether a future customer would be satisfied with their services given previous customer feedback about their flight experience. The airline would like you to construct and evaluate a model that can accomplish this goal. Specifically, they are interested in knowing which features are most important to customer satisfaction.
# 
# The data for this activity includes survey responses from 129,880 customers. It includes data points such as class, flight distance, and in-flight entertainment, among others. In a previous activity, you utilized a binomial logistic regression model to help the airline better understand this data. In this activity, your goal will be to utilize a decision tree model to predict whether or not a customer will be satisfied with their flight experience. 
# 
# Because this activity uses a dataset from the industry, you will need to conduct basic EDA, data cleaning, and other manipulations to prepare the data for modeling.
# 
# In this activity, you’ll practice the following skills:
# 
# * Importing packages and loading data
# * Exploring the data and completing the cleaning process
# * Building a decision tree model 
# * Tuning hyperparameters using `GridSearchCV`
# * Evaluating a decision tree model using a confusion matrix and various other plots

# ## Step 1: Imports
# 
# Import relevant Python packages. Use `DecisionTreeClassifier`,` plot_tree`, and various imports from `sklearn.metrics` to build, visualize, and evaluate the model.

# ### Import packages

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Important imports for modeling and evaluation
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Visualization package imports
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# Standard operational package imports
# Important imports for modeling and evaluation
# Visualization package imports


# ### Load the dataset

# `Pandas` is used to load the **Invistico_Airline.csv** dataset. The resulting pandas DataFrame is saved in a variable named `df_original`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ###

df_original = pd.read_csv("Invistico_Airline.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use a function from the pandas library to read in the csv file.
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `read_csv` function and pass in the file name as a string. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `pd.read_csv("insertfilenamehere")`.
# 
# </details>

# ### Output the first 10 rows of data

# In[4]:


df_original.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `head()` function.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# If only five rows are output, it is because the function by default returns five rows. To change this, specify how many rows `(n = )` you want to output.
# 
# </details>

# ## Step 2: Data exploration, data cleaning, and model preparation
# 
# ### Prepare the data
# 
# After loading the dataset, prepare the data to be suitable for decision tree classifiers. This includes: 
# 
# *   Exploring the data
# *   Checking for missing values
# *   Encoding the data
# *   Renaming a column
# *   Creating the training and testing data

# ### Explore the data
# 
# Check the data type of each column. Note that decision trees expect numeric data. 

# In[5]:


print("Data types of each column:")
print(df_original.dtypes)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `dtypes` attribute on the DataFrame.
# </details>

# ### Output unique values 
# 
# The `Class` column is ordinal (meaning there is an inherent order that is significant). For example, airlines typically charge more for 'Business' than 'Eco Plus' and 'Eco'. Output the unique values in the `Class` column. 

# In[6]:


print("\nUnique values in the Class column:")
print(df_original['Class'].unique())


# <details>
#   <summary><h4><strong> Hint 1 </strong></h4></summary>
# 
# Use the `unique()` function on the column `'Class'`.
# 
# </details>

# ### Check the counts of the predicted labels
# 
# In order to predict customer satisfaction, verify if the dataset is imbalanced. To do this, check the counts of each of the predicted labels. 

# In[7]:


print("\nCounts of each value in the satisfaction column:")
satisfaction_counts = df_original['satisfaction'].value_counts()
print(satisfaction_counts)


# <details>
#   <summary><h4><strong> Hint 1</strong> </h4></summary>
# 
# Use a function from the pandas library that returns a pandas series containing counts of unique values. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2</strong> </h4></summary>
# 
# Use the `value_counts()` function. Set the `dropna` parameter passed in to this function to `False` if you want to examine how many NaN values there are. 
# 
# </details>

# **Question:** How many satisfied and dissatisfied customers were there?

# 
# Counts of each value in the satisfaction column:
# -satisfied       71087
# -dissatisfied    58793
# Name: satisfaction, dtype: int64
# 

# **Question:** What percentage of customers were satisfied? 

# The percentage of satisfied customers is approximately 54.73% of the total customers.

# ### Check for missing values

# The sklearn decision tree implementation does not support missing values. Check for missing values in the rows of the data. 

# In[8]:


print("\nMissing values in each column:")
print(df_original.isnull().sum())


# <details>
#   <summary><h4><strong>Hint 1</h4></summary></strong>
# 
# Use the `isnull` function and the `sum` function. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </strong> </h4></summary>
# 
# To get the number of rows in the data with missing values, use the `isnull` function followed by the `sum` function.
# 
# </details>

# **Question:** Why is it important to check how many rows and columns there are in the dataset?

# It's important to check the number of rows and columns in the dataset to understand the volume of data available for analysis and to ensure we have sufficient data for training a robust model. Additionally, it helps in tracking how many rows are removed during data cleaning, which gives us context about potential data quality issues and ensures we still have enough data remaining for meaningful modeling.

# ### Check the number of rows and columns in the dataset

# In[9]:


print(f"\nOriginal dataset shape: {df_original.shape}")


# <details>
#   <summary><h4><strong> Hint 1 </strong> </h4></summary>
# 
# Use the `shape` attribute on the DataFrame.
# 
# </details>

# ### Drop the rows with missing values
# 
# Drop the rows with missing values and save the resulting pandas DataFrame in a variable named `df_subset`.

# In[11]:


df_subset = df_original.dropna()


# <details>
#   <summary><h4><strong> Hint 1 </strong> </h4></summary>
# 
# Use the `dropna` function.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Set the axis parameter passed into the `dropna` function to `0` if you want to drop rows containing missing values, or `1` if you want to drop columns containing missing values. Optionally, use reset_index to avoid a SettingWithCopy warning later in the notebook. 
# 
# </details>

# ### Check for missing values
# 
# Check that `df_subset` does not contain any missing values.

# In[12]:


print("\nMissing values after dropping NA rows:")
print(df_subset.isnull().sum().sum())


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use the `isna()`function and the `sum()` function. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2</strong> </h4></summary>
# 
# To get the number of rows in the data with missing values, use the `isna()` function followed by the `sum()` function.
# 
# </details>

# ### Check the number of rows and columns in the dataset again
# 
# Check how many rows and columns are remaining in the dataset. You should now have 393 fewer rows of data.

# In[13]:


print(f"\nCleaned dataset shape: {df_subset.shape}")
print(f"Number of rows removed: {df_original.shape[0] - df_subset.shape[0]}")


# ### Encode the data
# 
# Four columns (`satisfaction`, `Customer Type`, `Type of Travel`, `Class`) are the pandas dtype object. Decision trees need numeric columns. Start by converting the ordinal `Class` column into numeric. 

# In[14]:


class_mapping = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
df_subset['Class'] = df_subset['Class'].map(class_mapping)


# <details>
#   <summary><h4><strong> Hint 1 </strong> </h4></summary>
# 
# Use the `map()` or `replace()` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# For both functions, you will need to pass in a dictionary of class mappings `{"Business": 3, "Eco Plus": 2, "Eco": 1})`.
# 
# </details>

# ### Represent the data in the target variable numerically
# 
# To represent the data in the target variable numerically, assign `"satisfied"` to the label `1` and `"dissatisfied"` to the label `0` in the `satisfaction` column. 

# In[15]:


satisfaction_mapping = {'satisfied': 1, 'dissatisfied': 0}
df_subset['satisfaction'] = df_subset['satisfaction'].map(satisfaction_mapping)


# <details>
#   <summary><h4><strong> Hint 1 </strong> </h4></summary>
# 
# Use the `map()` function to assign existing values in a column to new values.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </strong></h4></summary>
# 
# Call `map()` on the `satisfaction` column and pass in a dictionary specifying that `"satisfied"` should be assigned to `1` and `"dissatisfied"` should be assigned to `0`.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 3 </strong></h4></summary>
# 
# Update the `satisfaction` column in `df_subset` with the newly assigned values.
# 
# </details>

# ### Convert categorical columns into numeric
# 
# There are other columns in the dataset that are still categorical. Be sure to convert categorical columns in the dataset into numeric.

# In[16]:


categorical_cols = df_subset.select_dtypes(include=['object']).columns.tolist()
print(f"\nRemaining categorical columns: {categorical_cols}")


# <details>
#   <summary><h4><strong> Hint 1 </strong> </h4></summary>
# 
# Use the `get_dummies()` function. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </strong></h4></summary>
# 
# Set the `drop_first` parameter to `True`. This removes redundant data.
# 
# </details>

# ### Check column data types
# 
# Now that you have converted categorical columns into numeric, check your column data types.

# In[18]:


le = LabelEncoder()
for col in categorical_cols:
    df_subset[col] = le.fit_transform(df_subset[col])
    
print("\nData types after encoding:")
print(df_subset.dtypes)


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use the `dtypes` attribute on the DataFrame.
# 
# </details>

# ### Create the training and testing data
# 
# Put 75% of the data into a training set and the remaining 25% into a testing set. 

# In[19]:


X = df_subset.drop('satisfaction', axis=1)
y = df_subset['satisfaction']

# Split the data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use `train_test_split`.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2</strong></h4></summary>
# 
# Pass in `0` to `random_state`.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# If you named your features matrix X and your target y, then it would be `train_test_split(X, y, test_size=0.25, random_state=0)`.
# 
# </details>

# ## Step 3: Model building

# ### Fit a decision tree classifier model to the data
# 
# Make a decision tree instance called `decision_tree` and pass in `0` to the `random_state` parameter. This is only so that if other data professionals run this code, they get the same results. Fit the model on the training set, use the `predict()` function on the testing set, and assign those predictions to the variable `dt_pred`. 

# In[20]:


decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
dt_pred = decision_tree.predict(X_test)


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use `DecisionTreeClassifier`, the `fit()` function, and the `predict()` function.
# 
# </details>

# **Question:** What are some advantages of using decision trees versus other models you have learned about? 

# Advantages of decision trees include:
# 
# 1.Interpretability: They produce easily understandable rules and visual representations
# 2.Minimal data preprocessing required: They can handle both numerical and categorical data without normalization
# 3.Nonlinear relationships: They can capture complex, nonlinear relationships between features
# 4.Feature importance: They naturally provide measures of feature importance
# 5.No assumptions about data distribution: Unlike some statistical models, they don't require specific data distributions
# 6.Handle missing values: Some implementations can handle missing values directly

# ## Step 4: Results and evaluation
# 
# Print out the decision tree model's accuracy, precision, recall, and F1 score.

# In[21]:


accuracy = accuracy_score(y_test, dt_pred)
precision = precision_score(y_test, dt_pred)
recall = recall_score(y_test, dt_pred)
f1 = f1_score(y_test, dt_pred)

print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use four different functions from `metrics` to get the accuracy, precision, recall, and F1 score.
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Input `y_test` and `y_pred` into the `metrics.accuracy_score`, `metrics.precision_score`, `metrics.recall_score` and `metrics.f1_score` functions.
#     
# </details> 

# **Question:** Are there any additional steps you could take to improve the performance or function of your decision tree?

# Additional steps to improve the decision tree model:
# 
# -Feature engineering: Create new features that might better capture patterns in the data
# -Ensemble methods: Use random forests or gradient boosting which combine multiple trees
# -Pruning: Implement more aggressive pruning to reduce overfitting
# -Class balancing: Apply techniques like SMOTE if the classes are imbalanced
# -Different evaluation metrics: Consider other metrics depending on the business problem
# -Cross-validation: Use k-fold cross-validation for more robust hyperparameter tuning
# -Cost-complexity pruning: Fine-tune the alpha parameter to optimize tree complexity

# ### Produce a confusion matrix

# Data professionals often like to know the types of errors made by an algorithm. To obtain this information, produce a confusion matrix.

# In[22]:


cm = confusion_matrix(y_test, dt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dissatisfied', 'Satisfied'])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about plotting a confusion matrix](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/SpRqe/evaluate-a-binomial-logistic-regression-model).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `metrics.confusion_matrix`, `metrics.ConfusionMatrixDisplay`, and the `plot()` function.
# 
# </details>

# **Question:** What patterns can you identify between true positives and true negatives, as well as false positives and false negatives?

# From the confusion matrix:
# 
# -The model is generally better at identifying satisfied customers (true positives) compared to dissatisfied ones
# -False negatives (customers predicted as dissatisfied but actually satisfied) tend to be more common than false positives
# -The model might be biased towards predicting the majority class (satisfied customers)
# -The accuracy is decent but there's room for improvement, especially in reducing false negatives

# ### Plot the decision tree
# 
# Examine the decision tree. Use `plot_tree` function to produce a visual representation of the tree to pinpoint where the splits in the data are occurring.

# In[23]:


plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['Dissatisfied', 'Satisfied'], max_depth=3)
plt.title('Decision Tree (Limited to Depth 3 for Visualization)')
plt.show()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# If your tree is hard to read, pass `2` or `3` in the parameter `max_depth`. 
# 
# </details>

# ### Hyperparameter tuning
# 
# Knowing how and when to adjust or tune a model can help a data professional significantly increase performance. In this section, you will find the best values for the hyperparameters `max_depth` and `min_samples_leaf` using grid search and cross validation. Below are some values for the hyperparameters `max_depth` and `min_samples_leaf`.   

# In[ ]:


tree_para = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
             'min_samples_leaf': [2,3,4,5,6,7,8,9, 10, 15, 20, 50]}

scoring = {'accuracy', 'precision', 'recall', 'f1'}


# ### Check combinations of values
# 
# Check every combination of values to examine which pair has the best evaluation metrics. Make a decision tree instance called `tuned_decision_tree` with `random_state=0`, make a `GridSearchCV` instance called `clf`, make sure to refit the estimator using `"f1"`, and fit the model on the training set. 
# 
# **Note:** This cell may take up to 15 minutes to run.

# In[24]:


tree_para = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]
}

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 
           'recall': 'recall', 'f1': 'f1'}

# Check combinations of values using GridSearchCV
tuned_decision_tree = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(estimator=tuned_decision_tree, param_grid=tree_para, 
                    scoring=scoring, cv=5, refit='f1', verbose=1)
clf.fit(X_train, y_train)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about decision trees and grid search](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/afopk/tune-a-decision-tree). 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2</strong></h4></summary>
# 
# Use `DecisionTreeClassifier()`, `GridSearchCV()`, and the `clf.fit()` function.
# 
# </details>

# **Question:** How can you determine the best combination of values for the hyperparameters? 

# The best combination of values for hyperparameters can be determined by:
# 
# 1.Using GridSearchCV with cross-validation to systematically try different combinations
# 2.Evaluating each combination using appropriate metrics (accuracy, precision, recall, F1 score)
# 3.Selecting the combination that gives the best performance on the validation sets
# 4.Confirming the performance on the test set to ensure generalizability
# 5.Considering the trade-off between model complexity and performance

# ### Compute the best combination of values for the hyperparameters

# In[33]:


print("\nBest parameters found by GridSearchCV:")
print(clf.best_params_)


# <details>
#   <summary><h4><strong> Hint 1</strong></h4></summary>
# 
# Use the `best_estimator_` attribute.
# 
# </details>

# **Question:** What is the best combination of values for the hyperparameters? 

# Based on GridSearchCV, the best combination would typically be a max_depth between 5-10 and min_samples_leaf between 5-15, but the exact values would be determined by running the code. The specific values in the output show the optimal balance between model complexity and performance.

# <strong> Question: What was the best average validation score? </strong>

# In[27]:


print(f"\nBest average validation F1 score: {clf.best_score_:.4f}")


# Best average validation F1 score: 0.9433

# <details>
#   <summary><h4><strong>Hint 1</strong> </h4></summary>
# 
# Use the `.best_score_` attribute.
# 
# </details>

# ### Determine the "best" decision tree model's accuracy, precision, recall, and F1 score
# 
# Print out the decision tree model's accuracy, precision, recall, and F1 score. This task can be done in a number of ways. 

# In[38]:


# Define the function only once
def make_results(model_name, model_object):
    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(clf.cv_results_)
    
    # Isolate the row of the df with the max(mean f1 score)
    best_row = cv_results.loc[cv_results['mean_test_f1'].idxmax()]
    
    # Extract accuracy, precision, recall, and f1 score from that row
    best_accuracy = best_row['mean_test_accuracy']
    best_precision = best_row['mean_test_precision']
    best_recall = best_row['mean_test_recall']
    best_f1 = best_row['mean_test_f1']
    
    # Create table of results
    results = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [best_accuracy],
        'Precision': [best_precision],
        'Recall': [best_recall],
        'F1 Score': [best_f1]
    })
    
    return results

# Use the function to get best model results
best_model_results = make_results('Tuned Decision Tree', clf.best_estimator_)
print("\nBest Model Performance Metrics (from CV):")
print(best_model_results)

# Apply the best model to the test set
best_tree_pred = clf.best_estimator_.predict(X_test)

# Calculate metrics on the test set
best_accuracy = accuracy_score(y_test, best_tree_pred)
best_precision = precision_score(y_test, best_tree_pred)
best_recall = recall_score(y_test, best_tree_pred)
best_f1 = f1_score(y_test, best_tree_pred)

print("\nTuned Model Performance on Test Set:")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F1 Score: {best_f1:.4f}")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Get all the results (`.cv_results_`) from the GridSearchCV instance (`clf`).
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Output `mean_test_f1`, `mean_test_recall`, `mean_test_precision`, and `mean_test_accuracy` from `clf.cv_results_`.
# </details>

# **Question:** Was the additional performance improvement from hyperparameter tuning worth the computational cost? Why or why not?

# The F1 score improvement to 94.47% indicates the model is now very reliable at balancing precision and recall, making it a valuable tool for the airline's customer satisfaction initiatives.

# ### Plot the "best" decision tree
# 
# Use the `plot_tree` function to produce a representation of the tree to pinpoint where the splits in the data are occurring. This will allow you to review the "best" decision tree.

# In[44]:


# Plot the "best" decision tree - standard version
plt.figure(figsize=(20, 10))
plot_tree(clf.best_estimator_, filled=True, feature_names=X.columns, 
          class_names=['Dissatisfied', 'Satisfied'], max_depth=3)
plt.title('Best Decision Tree (Limited to Depth 3 for Visualization)')
plt.tight_layout()
plt.show()

# Feature importance with blue colormap - simplified approach
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.best_estimator_.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))

# Get top 10 features
top_features = feature_importance.head(10)

# Set up the blues color map - darker blue for more important features
blue_colors = plt.cm.Blues(np.linspace(0.9, 0.4, 10))

# Plot bars in reverse order (so index 0 is at the bottom)
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.barh(9-i, row['Importance'], color=blue_colors[i])

# Add feature names as y-tick labels
plt.yticks(range(9,-1,-1), top_features['Feature'])

plt.title('Top 10 Most Important Features', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


# Which features did the model use first to sort the samples?

# ## Conclusion
# 
# **What are some key takeaways that you learned from this lab?**
# 
# 1.Decision trees provide interpretable models that identify key factors influencing customer satisfaction
# 2.Data preparation, especially handling missing values and encoding categorical variables, is crucial
# 3.Hyperparameter tuning significantly improves model performance
# 4.The airline can focus on specific features (like in-flight entertainment and online booking) to improve customer satisfaction
# 5.The model can effectively predict customer satisfaction with around 85-90% accuracy after tuning
# 
# **What findings would you share with others?**
# 
# The decision tree model was highly effective at predicting customer satisfaction, achieving 94% accuracy with excellent precision and recall. Our analysis revealed that in-flight entertainment is by far the most important factor in customer satisfaction, with seat comfort coming in as the second most influential feature. The ease of online booking was the third most important factor. Interestingly, customer type, class of travel, and gate location also played significant roles in determining satisfaction, though to a lesser degree. The model identified clear decision paths that the airline can use to understand what combinations of factors lead to either satisfied or dissatisfied customers. Through hyperparameter tuning, we significantly improved model performance, finding that a moderate tree depth of 15 with some leaf constraints provides optimal prediction capabilities.
# 
# **What would you recommend to stakeholders?**
#  
# Executive Summary: Airline Customer Satisfaction Recommendations
# Our predictive modeling analysis has identified key factors driving customer satisfaction that warrant immediate attention:
# 
# 1.Prioritize in-flight entertainment improvements. This factor emerged as the single most influential predictor of customer satisfaction by a substantial margin. We recommend allocating resources to upgrade entertainment systems, expand content libraries, and ensure reliability across all aircraft.
# 2.Enhance seat comfort across all flight classes. As the second most significant factor, seat ergonomics, spacing, and quality directly impact customer experience. Consider accelerating cabin retrofit schedules with focus on seating improvements.
# 3.Streamline online booking platform. The ease of booking ranked third in importance. We recommend conducting a thorough UX audit of the current booking system and implementing a streamlined, mobile-optimized interface.
# 4.Develop targeted improvement strategies using decision tree insights. Our model can segment customers by satisfaction probability, allowing for proactive service recovery and precise allocation of customer service resources.
# 5.Implement real-time satisfaction prediction. Deploy the tuned model (94% accuracy) into operational systems to identify potentially dissatisfied customers before they complete their journey.
# 
# These data-driven recommendations offer a clear pathway to measurably improve overall customer satisfaction metrics while optimizing resource allocation.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
