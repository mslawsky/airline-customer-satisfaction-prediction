#!/usr/bin/env python
# coding: utf-8

# # Activity: Build an XGBoost model

# ## Introduction
# 
# In this activity, you’ll build on the skills and techniques you learned in the decision tree and random forest lessons to construct your own XGBoost classification model. The XGBoost model is a very powerful extension of decision trees, so having a strong working familiarity with this process will strengthen your skills and resume as a data professional.
# 
# This activity is a continuation of the airlines project in which you built decision tree and random forest models. You will use the same data, but this time you will train, tune, and evaluate an XGBoost model. You’ll then compare the performance of all three models and decide which model is best. Finally, you’ll explore the feature importances of your model and identify the features that most contribute to customer satisfaction.
# 

# ## Step 1: Imports

# ### Import packages
# 
# Begin with your import statements. First, import `pandas`, `numpy`, and `matplotlib` for data preparation. Next, import scikit-learn (`sklearn`) for model preparation and evaluation. Then, import `xgboost`, which provides the classification algorithm you'll implement to formulate your predictive model.

# In[3]:


# Import relevant libraries and modules.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# For model preparation and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Import XGBoost
import xgboost as xgb


# ### Load the dataset
# 
# To formulate your model, `pandas` is used to import a csv of airline passenger satisfaction data called `Invistico_Airline.csv`. This DataFrame is called `airline_data`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[4]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

### YOUR CODE HERE ###

airline_data = pd.read_csv('Invistico_Airline.csv', error_bad_lines=False)


# ### Display the data

# Examine the first 10 rows of data to familiarize yourself with the dataset.

# In[5]:


# Display the first ten rows of data.

print(airline_data.head(10))


# ### Display the data type for each column
# 
# Next, observe the types of data present within this dataset.

# In[6]:


# Display the data type for each column in your DataFrame.

print(airline_data.dtypes)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall the methods for exploring DataFrames.
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Recall a property of a `pandas` DataFrame that allows you to view the data type for each column.</details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `.dtypes` on your DataFrame `airline_data` to view the data type of each column.</details>

# **Question:** Identify the target (or predicted) variable for passenger satisfaction. What is your initial hypothesis about which variables will be valuable in predicting satisfaction?

# The target (predicted) variable for passenger satisfaction is the "satisfaction" column, which appears to be categorical (object type). This would need to be encoded for modeling.
# Initial hypothesis about valuable predictor variables:
# 
# I hypothesize that the following variables will be most valuable in predicting passenger satisfaction:
# 
# 1.Service-related variables like "Seat comfort," "Food and drink," "On-board service," and "Inflight entertainment" - these directly impact the passenger experience
# 2.Delay-related variables ("Departure Delay in Minutes" and "Arrival Delay in Minutes") - delays typically have a strong negative impact on satisfaction
# 3.Class type - premium class passengers likely have different expectations and satisfaction thresholds
# 4."Flight Distance" - might influence expectations and tolerance for service quality
# Comfort factors like "Leg room service" and "Cleanliness"
# 
# I expect that delay times and service quality metrics will be the strongest predictors, while variables like "Gate location" might be less influential. Time-sensitive variables like "Departure/Arrival time convenient" may have moderate impact.

# ## Step 2: Model preparation
# 
# Before you proceed with modeling, consider which metrics you will ultimately want to leverage to evaluate your model.

# **Question:** Which metrics are most suited to evaluating this type of model?

# For evaluating a classification model predicting passenger satisfaction, the following metrics are most suitable:
# 
# 1.Accuracy - Provides an overall measure of correct predictions, useful for a balanced dataset
# 2.Precision - Measures how many passengers predicted as "satisfied" were actually satisfied, helping to avoid false positives
# 3.Recall - Measures how many actually satisfied passengers were correctly identified, helping to avoid false negatives
# 4.F1 Score - The harmonic mean of precision and recall, providing a balanced metric when class distribution might be uneven
# 5.Confusion Matrix - Visually represents true positives, false positives, true negatives, and false negatives, giving deeper insight into where the model succeeds or fails
# 
# If classes are imbalanced (e.g., many more dissatisfied than satisfied passengers or vice versa), accuracy alone could be misleading, making precision, recall, and F1 score particularly important. Given the business context of airline satisfaction, we might prioritize recall if identifying dissatisfied customers for intervention is the goal, or precision if targeting satisfied customers for loyalty programs.

# ### Prepare your data for predictions
# 
# You may have noticed when previewing your data that there are several non-numerical variables (`object` data types) within the dataset.
# 
# To prepare this DataFrame for modeling, first convert these variables into a numerical format.

# In[27]:


# Convert the object predictor variables to numerical dummies.

# Assuming airline_data is your DataFrame
# First, identify object predictor variables (excluding 'satisfaction')
object_columns = airline_data.select_dtypes(include=['object']).columns.tolist()

# Remove the target variable 'satisfaction' if it's in the list of object columns
object_columns.remove('satisfaction')  # Exclude 'satisfaction' from predictor variables

# Create dummy variables for object columns
airline_data_encoded = pd.get_dummies(airline_data, columns=object_columns, drop_first=True)

# Check the first few rows of the encoded data
print(airline_data_encoded.head())


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about feature engineering](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/5mEqu/introduction-to-feature-engineering).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `pandas` function for transforming categorical data into "dummy" variables.</details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `get_dummies()` function on your DataFrame `airline_data` to create dummies for the categorical variables in your dataset. Note that your target variable will also need this treatment.</details>

# ### Isolate your target and predictor variables
# Separately define the target variable (`satisfaction`) and the features.

# In[28]:


# Define the y (target) variable.

y = airline_data_encoded['satisfaction']
if pd.api.types.is_object_dtype(y):
    # Convert 'satisfaction' to numeric if it's categorical
    y = y.map({'satisfied': 1, 'neutral or dissatisfied': 0})  # Adjust mapping as needed

# Define the X (predictor) variables.

X = airline_data_encoded.drop('satisfaction', axis=1)

print(y.head())
print(X.head())


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data into x and y](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/ozK9K/build-a-decision-tree-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# In `pandas`, use square brackets `[]` to subset your DataFrame by specifying which column(s) to select. Also, quickly subset a DataFrame to exclude a particular column by using the `drop()` function and specifying the column to drop.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# In this case, your target variable was split into two columns from the dummy split. Be sure to include only the column which assigns a positive (i.e., "satisfied") outcome as 1.
# </details>

# ### Divide your data 
# 
# Divide your data into a training set (75% of the data) and test set (25% of the data). This is an important step in the process, as it allows you to reserve a part of the data that the model has not used to test how well the model generalizes (or performs) on new data.

# In[9]:


# Perform the split operation on your data.
# Assign the outputs as follows: X_train, X_test, y_train, y_test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about splitting your data between a training and test set](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/ozK9K/build-a-decision-tree-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# To perform the splitting, call the function in the `model_selection` module of `sklearn` on the features and target variable.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `train_test_split()` function, passing in both `features` and `target`, while configuring the appropriate `test_size`. Assign the output of this split as `X_train`, `X_test`, `y_train`, `y_test`.
# </details>

# ## Step 3: Model building

# ### "Instantiate" your XGBClassifer
# 
# Before you fit your model to your airline dataset, first create the XGB Classifier model and define its objective. You'll use this model to fit and score different hyperparameters during the GridSearch cross-validation process.

# In[10]:


# Define xgb to be your XGBClassifier.

xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about constructing a classifier model from `xgboost`](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/njRAP/build-an-xgboost-model-with-python).</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Note that the target variable in this case is a binary variable. </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use the `XGBClassifier()` from `xgboost`. Set the objective as `binary:logistic`.
# </details>

# ### Define the parameters for hyperparameter tuning
# 
# To identify suitable parameters for your `xgboost` model, first define the parameters for hyperparameter tuning. Specifically, consider tuning `max_depth`, `min_child_weight`, `learning_rate`, `n_estimators`, `subsample`, and/or `colsample_bytree`.
# 
# Consider a more limited range for each hyperparameter to allow for timely iteration and model training. For example, using a single possible value for each of the six hyperparameters listed above will take approximately one minute to run on this platform.
# 
# ```
# {
#     'max_depth': [4],
#     'min_child_weight': [3],
#     'learning_rate': [0.1],
#     'n_estimators': [5],
#     'subsample': [0.7],
#     'colsample_bytree': [0.7]
# }
# ```
# 
# If you add just one new option, for example by changing `max_depth: [4]` to `max_depth: [3, 6]`, and keep everything else the same, you can expect the run time to approximately double. If you use two possibilities for each hyperparameter, the run time would extend to ~1 hour. 
#          

# In[35]:


# Define parameters for tuning as `cv_params`.

cv_params = {
    'max_depth': [3, 6],
    'min_child_weight': [1, 3],
    'learning_rate': [0.1, 0.3],
    'n_estimators': [50, 100],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about hyperparameter tuning using GridSearch cross-validation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/afopk/tune-a-decision-tree).</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Consider a range of values for each parameter, similar to what you observed in the lesson. </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Define these parameters using a Python dictionary in the following format: `{'parameter1': [range,of,values]}`</details>

# **Question:** What is the likely effect of adding more estimators to your GridSearch?

# Adding more estimators (trees) to your GridSearch will have several significant effects:
# 
# 1.Increased computational time: Each additional estimator configuration multiplies the total combinations to be evaluated, leading to substantially longer training times.
# 2.Potential improvement in model performance: Up to a certain point, more estimators typically improve model accuracy and robustness by reducing variance through ensemble learning.
# 3.Diminishing returns: After a certain threshold, adding more estimators yields increasingly smaller performance gains while continuing to increase computational cost.
# 4.Risk of overfitting: Too many estimators might cause the model to overfit to the training data, especially with deeper trees or without proper regularization.
# 5.Memory usage: More estimators require more memory, which could become a limiting factor on your system.
# 
# The key is finding a balance where you have enough estimators for good performance without excessive computational burden or overfitting. For practical implementation, it's often useful to start with a smaller grid search and then refine around promising parameter areas.

# ### Define how the models will be evaluated
# 
# Define how the models will be evaluated for hyperparameter tuning. To yield the best understanding of model performance, utilize a suite of metrics.

# In[34]:


# Define your criteria as `scoring`.

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall what you've learned about [using metric evaluation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/afopk/tune-a-decision-tree) to determine the metrics you include.</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Consider what you've learned about the limitations of only including a single metric, such as `accuracy`. </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Define metrics which balance the false positives and false negatives in binary classification problems.

# ### Construct the GridSearch cross-validation 
# 
# Construct the GridSearch cross-validation using the model, parameters, and scoring metrics you defined. Additionally, define the number of folds and specify *which metric* from above will guide the refit strategy.

# In[29]:


# Construct your GridSearch.

from sklearn.model_selection import GridSearchCV

# Construct your GridSearch
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=cv_params,
    scoring='f1',  # Optimize for the 'f1' metric
    cv=5,
    refit='f1',  # Refitting based on the 'f1' metric
    verbose=1
)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall what you've learned about constructing a GridSearch for [cross-validation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/W4vAW/verify-performance-using-validation).</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Balance the time spent on validation with the number of folds you choose. </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Choose the refit method which simultaneously balances false positives and false negatives.

# ### Fit the GridSearch model to your training data
# 
# If your GridSearch takes too long, revisit the parameter ranges above and consider narrowing the range and reducing the number of estimators.
# 
# **Note:** The following cell might take several minutes to run.

# In[40]:


# fit the GridSearch model to training data

get_ipython().run_line_magic('time', 'grid_search.fit(X_train_simple, y_train_simple)')
    


# **Question:** Which optimal set of parameters did the GridSearch yield?

# Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 50, 'subsample': 0.7}
# Best F1 score: 0.9104857871765513

# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall what you've learned about the result of the GridSearch.</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Once you've fitted the GridSearch model to your training data, there will be an attribute to access which yields to the optimal parameter set.</details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Access the `best_params_` attribute from your fitted model. </details>

# ### Save your model for reference using `pickle`
# 
# Use the `pickle` library you've already imported to save the output of this model.

# In[41]:


print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Use `pickle` to save the trained model.

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about "pickling" prior models](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/FSnam/build-and-validate-a-random-forest-model-using-a-validation-data-set).</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The model to be pickled is the fitted GridSearch model from above. </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `pickle.dump()`, reference the fitted GridSearch model, and provide a name for the pickle file.

# ## Step 4: Results and evaluation
# 
# ### Formulate predictions on your test set
# 
# To evaluate the predictions yielded from your model, leverage a series of metrics and evaluation techniques from scikit-learn by examining the actual observed values in the test set relative to your model's prediction.
# 
# First, use your trained model to formulate predictions on your test set.

# In[42]:


# Apply your model to predict on your test data. Call this output "y_pred".

y_pred = grid_search.best_estimator_.predict(X_test_simple)

# Print evaluation metrics
print("Accuracy score:", accuracy_score(y_test_simple, y_pred))
print("Precision score:", precision_score(y_test_simple, y_pred))
print("Recall score:", recall_score(y_test_simple, y_pred))
print("F1 score:", f1_score(y_test_simple, y_pred))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall what you've learned about creating predictions from trained models.</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the fitted GridSearch model from your training set and predict the predictor variables you reserved in the train-test split.</details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `predict()` on your fitted model and reference `X_test` to create these predictions.
# </details>

# ### Leverage metrics to evaluate your model's performance
# 
# Apply a series of metrics from scikit-learn to assess your model. Specifically, print the accuracy score, precision score, recall score, and f1 score associated with your test data and predicted values.

# In[44]:


# Print evaluation metrics
print("Accuracy score:", accuracy_score(y_test_simple, y_pred))
print("Precision score:", precision_score(y_test_simple, y_pred))
print("Recall score:", recall_score(y_test_simple, y_pred))
print("F1 score:", f1_score(y_test_simple, y_pred))


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about model evaluation for detail on these metrics](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/njRAP/build-an-xgboost-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the function in the `metrics` module in `sklearn` to compute each of these metrics.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `accuracy_score()`, `precision_score()`, `recall_score()`, and `f1_score()`, passing `y_test` and `y_pred` into each.
# </details>

# **Question:** How should you interpret your accuracy score?

# The accuracy score of 0.9018 (90.18%) indicates that your XGBoost model correctly predicts passenger satisfaction about 90% of the time. This is a strong performance, showing that the model correctly classifies 9 out of 10 passengers as either satisfied or dissatisfied. In the airline industry context, this high accuracy suggests the model has captured meaningful patterns in the factors that influence customer satisfaction.

# **Question:** Is your accuracy score alone sufficient to evaluate your model?

# No, accuracy alone is not sufficient to evaluate the model. While 90.18% accuracy appears impressive, this metric can be misleading, especially if the classes are imbalanced. The dataset shows some imbalance (53,225 satisfied vs. 44,185 dissatisfied passengers), so relying solely on accuracy could hide potential issues with the model's performance on the minority class. Additional metrics like precision, recall, F1 score, and examining the confusion matrix provide a more comprehensive evaluation of the model's effectiveness.

# **Question:** When observing the precision and recall scores of your model, how do you interpret these values, and is one more accurate than the other?

# The precision score of 0.9119 indicates that when the model predicts a passenger is satisfied, it's correct about 91.2% of the time. The recall score of 0.9092 means the model successfully identifies 90.9% of all actually satisfied passengers. These values are very close, suggesting the model is well-balanced in its predictions.
# 
# Neither precision nor recall is inherently "more accurate" than the other - they measure different aspects of performance. Precision focuses on the quality of positive predictions (minimizing false positives), while recall emphasizes the model's ability to find all positive instances (minimizing false negatives). The importance of each depends on the business context. For airlines, a balance might be ideal, as both misidentifying dissatisfied customers as satisfied (precision error) and missing opportunities to recognize satisfied customers (recall error) have business implications.

# **Question:** What does your model's F1 score tell you, beyond what the other metrics provide?*

# The F1 score of 0.9106 represents the harmonic mean of precision and recall, providing a single metric that balances both concerns. This high F1 score (close to 1) indicates that the model maintains strong performance on both precision and recall simultaneously, without sacrificing one for the other.
# 
# The F1 score is particularly valuable because it penalizes extreme imbalances between precision and recall. In this case, the similar values for precision (0.9119) and recall (0.9092) result in an F1 score (0.9106) that's very close to both, confirming the model's balanced performance. This suggests the XGBoost model is robust and doesn't disproportionately favor one class over the other, making it reliable for overall customer satisfaction prediction in real-world airline scenarios.

# ### Gain clarity with the confusion matrix
# 
# Recall that a **confusion matrix** is a graphic that shows a model's true and false positives and true and false negatives. It helps to create a visual representation of the components feeding into the metrics above.
# 
# Create a confusion matrix based on your predicted values for the test set.

# In[47]:


# Construct and display your confusion matrix.

cm = confusion_matrix(y_test_simple, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dissatisfied', 'satisfied'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about model evaluation](https://www.coursera.org/learn/the-nuts-and-bolts-of-machine-learning/lecture/njRAP/build-an-xgboost-model-with-python).
# </details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the functions in the `metrics` module to create a confusion matrix.
# </details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call `confusion_matrix`, passing in `y_test` and `y_pred`. Next, utilize `ConfusionMatrixDisplay()` to display your confusion matrix.
# </details>

# **Question:** When observing your confusion matrix, what do you notice? Does this correlate to any of your other calculations?

# When examining the confusion matrix, several important patterns emerge:
# 
# 1.Strong diagonal performance: The matrix shows high values along the diagonal (top-left to bottom-right), indicating that the model correctly classifies most cases - approximately 13,000 dissatisfied passengers and 16,000 satisfied passengers.
# 2.Balanced errors: The off-diagonal elements (errors) are very similar - around 1,600 misclassifications in each direction. This indicates the model doesn't disproportionately favor one class over the other.
# 3.Confirmation of metrics: The balanced nature of the confusion matrix directly correlates with the nearly identical precision (0.9119) and recall (0.9092) scores we observed earlier. This explains why the F1 score (0.9106) is so close to both metrics.
# 4.Accuracy verification: We can calculate accuracy from the confusion matrix by dividing the sum of the diagonal elements (correct predictions) by the total observations. This matches our previously reported accuracy score of about 90.2%.
# 5.Class distribution: The matrix suggests the test set maintains a similar class imbalance as the training data, with more satisfied than dissatisfied passengers.
# 
# This visualization reinforces our earlier metric calculations and provides additional confidence that the XGBoost model is performing consistently well across both classes despite the slight class imbalance in the dataset.

# ### Visualize most important features
# 
# `xgboost` has a built-in function to visualize the relative importance of the features in the model using `matplotlib`. Output and examine the feature importance of your model.

# In[48]:


# Plot the relative feature importance of the predictor variables in your model.

plt.figure(figsize=(10, 8))
xgb.plot_importance(best_model, max_num_features=15)
plt.title('Feature Importance')
plt.show()


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Recall the attributes that are provided once the model is fitted to training data.</details>

# <details>
# <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Examine the `best_estimator_` attribute of your fitted model.</details>

# <details>
# <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# To easily visualize feature importance, call the built-in `plot_importance` function `xgboost` offers on the `best_estimator_`.</details>

# **Question:** Examine the feature importances outputted above. What is your assessment of the result? Did anything surprise you?

# The feature importance plot reveals several interesting insights about what drives airline customer satisfaction:
# 
# 1.Seat comfort dominance: The overwhelming importance of seat comfort (83.0) is notable and somewhat surprising. It's more than twice as influential as the second most important feature, suggesting physical comfort is the primary driver of passenger satisfaction.
# 2.Shift from previous models: This model shows a significant shift from previous decision tree and random forest models where inflight entertainment was identified as the #1 predictor. Now, inflight entertainment ranks second (36.0), but with less than half the importance of seat comfort. This substantial difference between model algorithms highlights how XGBoost has captured a different relationship between features and satisfaction.
# 3.Service and convenience features: Ease of online booking, food and drink, and customer loyalty status cluster as the next tier of importance (18-20 points each), indicating service quality and convenience are secondary satisfaction drivers.
# 4.Lower impact of operational factors: Surprisingly, factors like baggage handling, cleanliness, and online boarding have relatively low importance scores (7-8 points), despite often being emphasized in airline marketing.
# 5.Traveler demographics matter: The appearance of customer type (loyal customer) and travel type (business travel) among the important features suggests that satisfaction is partly influenced by who is traveling and why, not just service quality alone.
# 
# What's most surprising is the dramatic difference in relative importance between seat comfort and all other factors. This suggests that airlines looking to maximize customer satisfaction might achieve the greatest returns by investing in improved seating rather than entertainment systems or food service. The results also challenge conventional wisdom about what passengers value most during their travel experience.

# ### Compare models
# 
# Create a table of results to compare model performance.

# In[50]:


# Create a table of results to compare model performance.

model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.7989, 0.9433, 0.9387, 0.9018],  # From each model's results
    'Precision': [0.8160, 0.9444, 0.9348, 0.9119], 
    'Recall': [0.8215, 0.9435, 0.9464, 0.9092],
    'F1 Score': [0.8187, 0.9439, 0.9406, 0.9106]
})
print(model_comparison)


# <details>
# <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Create a DataFrame and using the `pd.DataFrame()` function. 
# 
# </details>

# **Question:** How does this model compare to the decision tree and random forest models you built in previous labs? 

# The XGBoost model achieved good performance with an accuracy of 90.18%, precision of 91.19%, recall of 90.92%, and an F1 score of 91.06%. While these are strong metrics, the XGBoost model didn't outperform the decision tree or random forest models in this particular case.
# 
# The decision tree model performed best overall with the highest accuracy (94.33%), precision (94.44%), and F1 score (94.39%). The random forest model was a close second, with slightly lower accuracy and precision but the highest recall (94.64%) among all models.
# 
# All three ensemble/tree-based models (decision tree, random forest, and XGBoost) substantially outperformed the baseline logistic regression model, which confirms that the relationship between the features and customer satisfaction is complex and non-linear.
# 
# The fact that the simpler decision tree performed better than XGBoost in this case is interesting. This could be due to:
# 
# -The dataset structure might be particularly well-suited to decision tree splits
# -The hyperparameters for XGBoost might not have been fully optimized
# -The decision tree might be overfitting to the training data (though the high performance on test data suggests this may not be a major issue)
# 
# XGBoost typically excels when there's more complex, noisy data where its gradient boosting approach can iteratively improve predictions. In this case, the patterns in the airline satisfaction data might be captured sufficiently well by a single well-tuned decision tree.
# 
# For production use, I would recommend considering the random forest model, as it offers a good balance of high performance (94.06% F1 score) with potentially better generalization properties than a single decision tree, though all tree-based models showed strong results that would be valuable for predicting customer satisfaction.
# 

# ## Considerations
# 
# **What are some key takeaways you learned from this lab?**
# 
# The XGBoost modeling exercise provided several valuable insights:
# 
# 1.Model Performance Hierarchy: While XGBoost delivered strong results (90.18% accuracy), surprisingly the decision tree model performed best (94.33% accuracy), followed by random forest (93.87% accuracy). This challenges the common assumption that more complex models always yield better results.
# 2.Feature Importance Clarity: XGBoost effectively identified the most influential factors in passenger satisfaction, with seat comfort emerging as the dominant factor, followed by inflight entertainment. This differs from previous models that ranked inflight entertainment highest.
# 3.Data Preparation Criticality: The importance of proper data preprocessing was highlighted, particularly regarding encoding categorical variables and handling missing values, which significantly impact model performance.
# 4.Hyperparameter Tuning Impact: Grid search cross-validation was essential for optimizing model performance, showing how carefully tuned parameters can enhance predictive accuracy.
# 5.Multi-Model Comparison Value: Comparing different modeling approaches (logistic regression, decision tree, random forest, and XGBoost) provided a more comprehensive understanding of the data relationships than any single model could offer.
# 
# 
# 
# **How would you share your findings with your team?**
# 
# I would share these findings with my team through:
# 
# 1.A concise presentation highlighting:
# 
# -Side-by-side performance metrics of all models
# -Feature importance visualization from XGBoost
# -Interactive demonstration of how different hyperparameter values affected model performance
# -Key limitations and remaining challenges
# 
# 
# 2.Code walkthrough focusing on:
# 
# -Preprocessing techniques that proved most effective
# -Implementation of GridSearchCV for hyperparameter tuning
# -How we addressed data quality issues
# -Best practices for model evaluation
# 
# 
# 3.Collaborative documentation including:
# 
# -Technical approach with decision points and rationales
# -Python notebook with detailed comments for reproducibility
# -Summary of lessons learned and recommendations for future modeling efforts
# 
# **What would you share with and recommend to stakeholders?**
# 
# Executive Summary: Airline Customer Satisfaction Prediction
# 
# FINDINGS & RECOMMENDATIONS
# Our comprehensive analysis of airline customer satisfaction data using advanced machine learning techniques has delivered actionable intelligence that can directly impact business performance.
# 
# Key Insights:
# 
# • Physical comfort aspects, particularly seat comfort, overwhelmingly drive customer satisfaction (83% of predictive importance)
# • In-flight entertainment quality is the second most influential factor (36%)
# • Ease of online booking ranks third in importance (20%)
# • Customer loyalty status and convenient departure/arrival times also significantly influence satisfaction
# 
# Model Performance:
# Our XGBoost model achieved 90.2% accuracy in predicting customer satisfaction, providing reliable guidance for decision-making. While slightly less accurate than our decision tree model (94.3%), it offers more nuanced insights into feature relationships.
# 
# Strategic Recommendations:
# 
# 1.Prioritize Cabin Upgrades: Invest in seat comfort improvements as the primary driver of satisfaction, especially in economy class where ROI potential is highest.
# 2.Enhance Entertainment Systems: Modernize in-flight entertainment options focusing on content variety and system responsiveness, the second most influential factor.
# 3.Streamline Digital Experience: Optimize the online booking platform to capitalize on its strong impact on overall satisfaction.
# 4.Implement Targeted Interventions: Deploy predictive analytics to identify at-risk passengers for proactive service recovery during their journey.
# 5.Develop Measurement Framework: Establish KPIs specifically tracking improvements in top satisfaction drivers to measure initiative success.
# 
# By focusing resources on these data-validated priorities, we project significant improvements in satisfaction scores within 6-12 months, with corresponding positive impacts on loyalty, repeat business, and revenue performance.
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
