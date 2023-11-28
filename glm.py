# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset (full path file to avoid errors)
df = pd.read_csv("path")

# Remove the "id" column 
df = df.drop(columns=["id"])

# Remove the nans
df = df.dropna()

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=["gender", "married", "hypertension", "heart_disease", "occupation", "residence", "smoking_status"])

# Define the independent and dependent variables
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Fit the GLM model with a binomial distribution
glm_binom = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial())
result = glm_binom.fit()

# Print the summary of the model
print(result.summary())

# Predict the stroke variable based on the X values
y_pred = result.predict(sm.add_constant(X))
y_pred = [1 if i > 0.5 else 0 for i in y_pred]

# Print the confusion matrix and classification report
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
