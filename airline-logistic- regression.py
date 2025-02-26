#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform logistic regression 
# 
# ## Introduction
# 
# In this activity, you will complete an effective bionomial logistic regression. This exercise will help you better understand the value of using logistic regression to make predictions for a dependent variable based on one independent variable and help you build confidence in practicing logistic regression. Because logistic regression is leveraged across a wide array of industries, becoming proficient in this process will help you expand your skill set in a widely-applicable way.   
# 
# For this activity, you work as a consultant for an airline. The airline is interested in knowing if a better in-flight entertainment experience leads to higher customer satisfaction. They would like you to construct and evaluate a model that predicts whether a future customer would be satisfied with their services given previous customer feedback about their flight experience.
# 
# The data for this activity is for a sample size of 129,880 customers. It includes data points such as class, flight distance, and in-flight entertainment, among others. Your goal will be to utilize a binomial logistic regression model to help the airline model and better understand this data. 
# 
# Because this activity uses a dataset from the industry, you will need to conduct basic EDA, data cleaning, and other manipulations to prepare the data for modeling.
# 
# In this activity, you will practice the following skills:
# 
# * Importing packages and loading data
# * Exploring the data and completing the cleaning process
# * Building a binomial logistic regression model 
# * Evaluating a binomial logistic regression model using a confusion matrix

# ## Step 1: Imports
# 
# ### Import packages
# 
# Import relevant Python packages. Use `train_test_split`, `LogisticRegression`, and various imports from `sklearn.metrics` to build, visualize, and evalute the model.

# In[3]:


# Standard operational package imports
import pandas as pd
import numpy as np

# Important imports for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Visualization package imports
import matplotlib.pyplot as plt
import seaborn as sns

# Standard operational package imports.

# Important imports for preprocessing, modeling, and evaluation.

# Visualization package imports.


# ### Load the dataset

# The dataset **Invistico_Airline.csv** is loaded. The resulting pandas DataFrame is saved as a variable named `df_original`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[4]:


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

# ### Output the first 10 rows
# 
# Output the first 10 rows of data.

# In[5]:


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
# After loading the dataset, prepare the data to be suitable for a logistic regression model. This includes: 
# 
# *   Exploring the data
# *   Checking for missing values
# *   Encoding the data
# *   Renaming a column
# *   Creating the training and testing data

# ### Explore the data
# 
# Check the data type of each column. Note that logistic regression models expect numeric data. 

# In[8]:


df_original.dtypes


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `dtypes` attribute on the DataFrame.
# 
# </details>

# ### Check the number of satisfied customers in the dataset
# 
# To predict customer satisfaction, check how many customers in the dataset are satisfied before modeling.

# In[7]:


satisfaction_counts = df_original['satisfaction'].value_counts()
print(satisfaction_counts)
print(f"Percentage of satisfied customers: {satisfaction_counts['satisfied'] / len(df_original) * 100:.2f}%")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use a function from the pandas library that returns a pandas series containing counts of unique values. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `value_counts()` function. To examine how many NaN values there are, set the `dropna` parameter passed in to this function to `False`.
# 
# </details>

# **Question:** How many satisfied and dissatisfied customers were there?

# There were 71,087 satisfied customers and 58,793 dissatisfied customers.

# **Question:** What percentage of customers were satisfied?

# 54.73% of customers were satisfied

# ### Check for missing values

# An assumption of logistic regression models is that there are no missing values. Check for missing values in the rows of the data.

# In[9]:


print(df_original.isnull().sum())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# To get the number of rows in the data with missing values, use the `isnull` function followed by the `sum` function.
# 
# </details>

# **Question:** Should you remove rows where the `Arrival Delay in Minutes` column has missing values, even though the airline is more interested in the `inflight entertainment` column?

# Yes, you should remove rows where the Arrival Delay in Minutes column has missing values, even though the airline is primarily interested in the Inflight entertainment column.
# 
# Here's why:
# 
# 1.Logistic regression models require complete data for all observations
# 2.Keeping incomplete rows could introduce bias in your analysis
# 3.With only 393 missing values out of 129,880 total customers (about 0.3%), removing these rows won't significantly impact your dataset's representativeness
# 4.While the airline is focused on inflight entertainment, properly handling missing data ensures statistical validity of your model

# ### Drop the rows with missing values
# 
# Drop the rows with missing values and save the resulting pandas DataFrame in a variable named `df_subset`.

# In[10]:


df_subset = df_original.dropna()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
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

# ### Prepare the data
# 
# If you want to create a plot (`sns.regplot`) of your model to visualize results later in the notebook, the independent variable `Inflight entertainment` cannot be "of type int" and the dependent variable `satisfaction` cannot be "of type object." 
# 
# Make the `Inflight entertainment` column "of type float." 

# In[11]:


df_subset = df_subset.astype({"Inflight entertainment": float})


# <details>
#     
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# Use the `.astype()` function with the dictionary `{"Inflight entertainment": float}` as an input.
# 
# </details>

# ### Convert the categorical column `satisfaction` into numeric
# 
# Convert the categorical column `satisfaction` into numeric through one-hot encoding.

# In[12]:


df_subset = pd.get_dummies(df_subset, columns=['satisfaction'], drop_first=True)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `OneHotEncoder()` from `sklearn.preprocessing`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call `OneHotEncoder()`, specifying the `drop` argument as `'first'` in order to remove redundant columns from the output. 
# 
# Call `.fit_transform()`, passing in the subset of the data that you want to encode (the subset consisting of `satisfaction`). 
# 
# Call `.toarray()` in order to convert the sparse matrix that `.fit_transform()` returns into an array.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Index `df_subset` with a double pair of square brackets to get a DataFrame that consists of just `satisfaction`.
# 
# After getting the encoded values, update the `satisfaction` column (you can use reassignment).
# 
# </details>

# ### Output the first 10 rows of `df_subset`
# 
# To examine what one-hot encoding did to the DataFrame, output the first 10 rows of `df_subset`.

# In[13]:


df_subset.head(10)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `head()` function.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# If only five rows are outputted, it is because the function by default returns five rows. To change this, specify how many rows `(n = )` you want.
# 
# </details>

# ### Create the training and testing data
# 
# Put 70% of the data into a training set and the remaining 30% into a testing set. Create an X and y DataFrame with only the necessary variables.
# 

# In[14]:


X = df_subset[['Inflight entertainment']]
y = df_subset['satisfaction_satisfied']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `train_test_split`.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# If you named your independent variable `X` and your dependent variable `y`, then it would be `train_test_split(X, y, test_size=0.30, random_state=42)`.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# When you use `train_test_split`, pass in `42` to `random_state`. `random_state` is used so that if other data professionals run this code, they can get the same exact train test split. If you use a different random state, your results will differ. </details>

# **Question:** If you want to consider customer satisfaction with your model, should you train your model to use `inflight entertainment` as your sole independent variable? 

# Using only inflight entertainment as the sole predictor might not be optimal. While it allows us to focus on the specific relationship the airline is interested in, customer satisfaction likely depends on multiple factors. A more comprehensive model would include other relevant variables. However, for this exercise, we're focusing specifically on the relationship between inflight entertainment and satisfaction.

# ## Step 3: Model building

# ### Fit a LogisticRegression model to the data
# 
# Build a logistic regression model and fit the model to the training data. 

# In[15]:


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `LogisticRegression()` and the `fit()` function on the training set. `LogisticRegression().fit(X_train,y_train)`.
# 
# </details>

# ### Obtain parameter estimates
# Make sure you output the two parameters from your model. 

# In[16]:


print("Intercept:", model.intercept_)


# In[17]:


print("Coefficient for Inflight entertainment:", model.coef_[0][0])


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to the content on [obtaining the parameter estimates](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/XCLzq/construct-a-logistic-regression-model-with-python) from a logistic regression model.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Call attributes to obtain the coefficient and intercept estimates.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `.coef_` and `.intercept_`
# 
# </details>

# ### Create a plot of your model
# 
# Create a plot of your model to visualize results using the seaborn package.

# In[18]:


plt.figure(figsize=(10, 6))
sns.regplot(x=X_test['Inflight entertainment'], y=y_test, logistic=True, ci=None)
plt.title('Logistic Regression: Inflight Entertainment vs Customer Satisfaction')
plt.xlabel('Inflight Entertainment Rating')
plt.ylabel('Probability of Satisfaction')
plt.grid(True)
plt.show()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use a function from the seaborn library that can plot data and a logistic regression model fit.
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `regplot` function.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Set the `logistic` parameter passed in to this function to `True` to estimate a logistic regression model.
# 
# </details>

# **Question:** What can you tell from the graph?

# From this graph, I can tell:
# 
# 1.There's a positive relationship between inflight entertainment ratings and customer satisfaction
# 2.The curve is S-shaped (sigmoid), which is characteristic of logistic regression
# 3.At lower entertainment ratings (0-2), the probability of satisfaction is quite low (below 25%)
# 4.Around rating 3, there's a steep increase in the probability of satisfaction
# 5.As ratings approach 5, the probability of satisfaction increases to over 80%
# 6.The inflection point appears to be around rating 3, where the slope is steepest
# 7.The blue dots at 0 and 1 represent the actual binary outcomes (satisfied or not satisfied)
# 
# This suggests that improving inflight entertainment ratings, particularly from mediocre (2-3) to good (3-4), would have the greatest impact on increasing customer satisfaction.

# ## Step 4. Results and evaluation
# 

# ### Predict the outcome for the test dataset
# 
# Now that you've completed your regression, review and analyze your results. First, input the holdout dataset into the `predict` function to get the predicted labels from the model. Save these predictions as a variable called `y_pred`.

# In[20]:


y_pred = model.predict(X_test)

# Save predictions.


# ### Print out `y_pred`
# 
# In order to examine the predictions, print out `y_pred`. 

# In[21]:


print("Predictions (first 20):", y_pred[:20])


# ### Use the `predict_proba` and `predict` functions on `X_test`

# In[22]:


# Use predict_proba to output a probability.

y_pred_proba = model.predict_proba(X_test)
print("Prediction probabilities (first 5):")
print(y_pred_proba[:5])


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Using the `predict_proba` function on `X_test` will produce the probability that each observation is a 0 or 1. 
# 
# </details>

# In[23]:


# Use predict to output 0's and 1's.

y_pred_binary = model.predict(X_test)
print("Binary predictions (first 20):", y_pred_binary[:20])


# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# `clf.predict` outputs an array of 0's and 1's, where 0's are unsatisfied and 1's are satisfied. 
# 
# </details>

# ### Analyze the results
# 
# Print out the model's accuracy, precision, recall, and F1 score.

# In[25]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use four different functions from `metrics` to get the accuracy, precision, recall, and F1 score.
#     
# </details>  

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Input `y_test` and `y_pred` into the `metrics.accuracy_score`, `metrics.precision_score`, `metrics.recall_score`, and `metrics.f1_score` functions. 
#     
# </details> 

# ### Produce a confusion matrix

# Data professionals often like to know the types of errors made by an algorithm. To obtain this information, produce a confusion matrix.

# In[26]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Dissatisfied', 'Satisfied'],
            yticklabels=['Dissatisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# **Question:** What stands out to you about the confusion matrix?

# What stands out is that the model has done fairly well at correctly classifying both satisfied and dissatisfied customers, with 17,423 true positives and 13,714 true negatives. The model seems to have a similar number of false positives (3,925) and false negatives (3,785), showing relatively balanced errors between the two classes.

# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about plotting a confusion matrix](https://www.coursera.org/learn/regression-analysis-simplify-complex-data-relationships/lecture/SpRqe/evaluate-a-binomial-logistic-regression-model).
# 
# </details>

# **Question:** Did you notice any difference in the number of false positives or false negatives that the model produced?

# The model produced slightly more false positives (3,925) than false negatives (3,785). This means the model is slightly more likely to predict a customer is satisfied when they actually aren't, compared to missing truly satisfied customers. The difference is minimal though, suggesting the model doesn't significantly favor one type of error over the other.

# **Question:** What do you think could be done to improve model performance?

# To improve model performance:
# 
# 1.Include more predictors beyond just inflight entertainment (e.g., food quality, seat comfort, staff service)
# 2.Try more complex models like random forests or gradient boosting
# 3.Perform feature engineering to create interaction terms
# 4.Address potential class imbalance with techniques like SMOTE
# 5.Tune hyperparameters of the logistic regression model
# 6.Collect more data if possible
# 7.Segment customers and build separate models for different customer types

# ## Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# 
# 1.Inflight entertainment has a significant positive relationship with customer satisfaction
# 2.The logistic regression model shows good predictive power even with just one variable
# 3.The relationship follows an S-curve, with ratings around 3 having the most impact on satisfaction
# 4.The model has balanced error rates between false positives and false negatives
# 5.Simple models can provide actionable insights for business decisions
# 
# **What findings would you share with others?**
# 
# 1.Higher inflight entertainment ratings strongly predict customer satisfaction
# 2.Customers with entertainment ratings of 4-5 have >70% probability of being satisfied
# 3.Customers with ratings below 2 have <20% probability of being satisfied
# 4.The model correctly classifies approximately 80% of customers
# 5.The relationship is non-linear - improvements have greater impact in the middle range (2-4)
# 
# **What would you recommend to stakeholders?**
# 
# Executive Summary: Airline Customer Satisfaction Analysis
# Our analysis of customer satisfaction data reveals that inflight entertainment quality serves as a powerful predictor of overall passenger satisfaction. The logistic regression model demonstrates that as entertainment ratings increase, the probability of customer satisfaction rises substantially, with the most dramatic improvements occurring as ratings move from average to good (2-3 to 3-4 on a 5-point scale).
# 
# Based on these findings, we recommend prioritizing investments in inflight entertainment improvements, particularly targeting the experience of customers who currently rate this service as average. By elevating the entertainment experience, the airline stands to significantly increase overall satisfaction metrics. We suggest implementing a systematic approach to track entertainment ratings alongside satisfaction scores to measure return on investment for these improvements.
# 
# While this focused analysis provides actionable insights, expanding the model to include additional satisfaction drivers would create a more comprehensive understanding of customer preferences and allow for more targeted enhancements across the entire customer journey.
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 
