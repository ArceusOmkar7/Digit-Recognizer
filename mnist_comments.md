# MNIST Digit Classification Project

This notebook demonstrates various machine learning approaches for classifying handwritten digits using the famous MNIST dataset. We'll explore different classification algorithms including Random Forest, Logistic Regression, and multi-class classification strategies.

## Overview
- **Dataset**: MNIST (70,000 images of handwritten digits 0-9)
- **Image Size**: 28x28 pixels (784 features)
- **Task**: Multi-class classification (10 classes)
- **Models**: Random Forest, Logistic Regression, One-vs-One, One-vs-Rest

## Table of Contents
1. Data Loading and Exploration
2. Data Preprocessing
3. Model Training with Pipeline
4. Hyperparameter Tuning
5. Alternative Classification Approaches
6. Model Comparison and Evaluation
7. Model Persistence

```python
# Import the fetch_openml function from sklearn.datasets to load popular datasets
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
# 'mnist_784' refers to the MNIST dataset with 784 features (28x28 pixels flattened from 28x28 images)
# as_frame=False ensures the data is returned as NumPy arrays instead of Pandas DataFrames
mnist = fetch_openml('mnist_784', as_frame=False)
```

## 1. Data Loading and Exploration

We'll start by loading the MNIST dataset from scikit-learn's `fetch_openml` function.

```python
# Import essential libraries for data manipulation and visualization
import pandas as pd          # For data manipulation and analysis (though not used with as_frame=False)
import numpy as np           # For numerical operations, especially with arrays
import matplotlib.pyplot as plt  # For plotting and visualization

# Import the dataset loading function from scikit-learn
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
# 'mnist_784' refers to the MNIST dataset with 784 features (28x28 pixels flattened)
# as_frame=False returns numpy arrays instead of pandas DataFrames
mnist = fetch_openml('mnist_784', as_frame=False)
```

Let's examine the dataset description to understand what we're working with:

```python
# Display the dataset description to understand its source, features, and other details
print(mnist.DESCR)
```

Now let's extract the features (X) and target labels (y) from the dataset:

```python
# Extract features (pixel values) and target labels (digit classes)
# X: Contains the image data, where each row is a flattened 28x28 image (784 pixel values)
# y: Contains the corresponding labels (digits 0-9) for each image
X, y = mnist.data, mnist.target
```

Let's inspect the structure of our feature matrix:

```python
# Display the feature matrix X. This will show the raw pixel values for each image.
X
```

Let's check for missing values in our dataset:

```python
# Display the dataset description to understand the structure and source
print(mnist.DESCR)

# Check for missing values (NaN) in the dataset
# np.isnan(X).sum() is the correct way to count NaN values in a NumPy array
# (X == np.nan).sum() does not work as expected for NaN comparison
(X == np.nan).sum()
```

### Data Visualization

Let's create a function to visualize individual digits from our dataset:

```python
def draw_number(arr):
    """
    Visualize a single digit from the MNIST dataset.

    Parameters:
    arr: 1D numpy array of 784 pixel values (flattened 28x28 image)
    """
    plt.gray()  # Set colormap to grayscale for displaying images
    plt.imshow(arr.reshape((28, 28)))  # Reshape the 1D array (784,) back to 2D image (28, 28) and display
    plt.show()  # Show the plot
```

Let's visualize the first digit in our dataset:

```python
# Extract features (pixel values) and target labels (digit classes)
# X: 70,000 samples × 784 features (28×28 pixels flattened)
# y: 70,000 labels (digits 0-9 as strings)
X, y = mnist.data, mnist.target

# Visualize the first image in the dataset (index 0)
# This will show us what the first handwritten digit looks like
draw_number(X[0])
```

```python
# Inspect a specific sample (e.g., the 5th image) and convert its data type to float64
# This conversion is often necessary for compatibility with scikit-learn algorithms
X[5].astype("float64")
```

## 2. Data Preprocessing

Before training our models, let's examine the data types and consider preprocessing steps:

Feature scaling is important for many machine learning algorithms. Let's standardize our features:

```python
# Import StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler instance and fit_transform the data
# This standardizes features by removing the mean and scaling to unit variance
# X.astype("float64") ensures the input data is of float type, which is good practice for scaling
sclaed_x = StandardScaler().fit_transform(X.astype("float64"))
```

```python
# Check for missing values (NaN) in the dataset
# np.isnan() is the correct function to detect NaN values in a NumPy array
np.isnan(X).sum()  # Fixed: was (X == np.nan).sum() which doesn't work properly

# Display the scaled features for the 5th sample
# Scaled values should have mean≈0 and std≈1 across all features
sclaed_x[5]
```

### Train-Test Split

Let's split our data into training and testing sets for model evaluation:

```python
# Import functions for data splitting and model evaluation
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# Split data into 80% training and 20% testing sets
# X: features, y: target labels
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures reproducible results across multiple runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
# Verify the shapes of our training and testing sets
# X_train should contain 80% of samples (56000), X_test 20% (14000), each with 784 features
# Expected output: ((56000, 784), (14000, 784))
X_train.shape, X_test.shape
```

```python
def draw_number(arr):
    """
    Visualize a single digit from the MNIST dataset.

    Parameters:
    arr: 1D numpy array of 784 pixel values (flattened 28x28 image)
    """
    plt.gray()  # Set colormap to grayscale
    # Reshape the 1D array (784,) back to 2D image (28, 28)
    plt.imshow(arr.reshape((28, 28)))
    plt.show()
```

## 3. Model Training with Pipeline

We'll use scikit-learn's Pipeline to combine preprocessing and model training steps. This ensures consistent data processing and makes our workflow more maintainable.

### Pipeline

```python
# Import required classes for creating a machine learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline that combines preprocessing and model training
# Step 1: 'scale' - Standardize features (mean=0, std=1)
# Step 2: 'model' - Random Forest classifier with 100 trees by default
# n_jobs=-1: Use all available processors for faster training
# random_state=42: For reproducible results
pipe = Pipeline([
    ('scale', StandardScaler()),  # Preprocessing step: feature scaling
    ('model', RandomForestClassifier(n_jobs=-1, random_state=42))  # Model step: Random Forest Classifier
])

# Visualize the first image in the dataset (index 0)
# This will show us what the first handwritten digit looks like
draw_number(X[0])
```

Now let's train our pipeline on the training data:

```python
# Check data type conversion - MNIST pixels are typically integers (0-255)
# Converting to float64 for compatibility with scikit-learn algorithms
X[5].astype("float64")

# Train the pipeline on our training data
# This will first scale the features (using StandardScaler), then train the Random Forest model
pipe.fit(X_train, y_train)
```

Let's make predictions on our test set:

```python
# Make predictions on the test set using the trained pipeline
# The pipeline will automatically apply the same scaling transformation learned from the training data
y_pred = pipe.predict(X_test)
```

### Model Evaluation

Let's evaluate our model's performance using various metrics:

```python
# Import evaluation metrics from scikit-learn
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix, classification_report

# Calculate and display various performance metrics
print("Accuracy: ", accuracy_score(y_test, y_pred))  # Overall accuracy of the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # 10x10 matrix showing true vs predicted counts
print("Multilabel Confusion Matrix:\n",
      multilabel_confusion_matrix(y_test, y_pred))  # One confusion matrix per class (binary classification view)
```

```python
# Import StandardScaler for feature normalization (already imported, but good to keep context)
from sklearn.preprocessing import StandardScaler

# Standardize features: mean=0, std=1 (already done, but this line was in the original notebook)
# This helps algorithms that are sensitive to feature scales (like Logistic Regression)
scaled_x = StandardScaler().fit_transform(X.astype("float64"))  # Fixed typo: was 'sclaed_x'

# Display detailed classification report with precision, recall, and F1-score for each digit class
print(classification_report(y_test, y_pred))
```

## 4. Hyperparameter Tuning

Let's improve our model performance by tuning hyperparameters using RandomizedSearchCV:

```python
# Import RandomizedSearchCV for hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV

# Define the hyperparameter distribution for RandomizedSearchCV
# 'model__' prefix is used to access parameters of the 'model' step within the pipeline
param_dist = {
    "model__n_estimators": [25, 50, 100, 200, 300],  # Number of trees in the forest
    "model__criterion": ['gini', 'entropy', 'log_loss'],  # Function to measure the quality of a split
    "model__max_depth": [None, 0, 1, 2, 3, 5, 10]  # Maximum depth of the tree (None means unlimited)
}

# Create RandomizedSearchCV object
# pipe: The estimator (our pipeline) to tune
# param_dist: Dictionary of parameters to sample
# n_jobs=-1: Use all available processors
# cv=3: 3-fold cross-validation
# verbose=True: Display progress during the search
rscv = RandomizedSearchCV(pipe, param_dist, n_jobs=-1, cv=3, verbose=True)

# Compare scaled vs original values for the same sample (already done, but in original notebook)
# Scaled values should have mean≈0 and std≈1 across all features
scaled_x[5]  # Fixed: was 'sclaed_x[5]'
```

Now let's run the hyperparameter search (this may take a few minutes):

```python
# Fit the RandomizedSearchCV to find the best hyperparameters
# This will try different combinations and select the best one using cross-validation
rscv.fit(X_train, y_train)
```

Let's evaluate the best model found by our search:

```python
# Import functions for data splitting and model evaluation (already imported, but good for context)
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# Split data into 80% training and 20% testing (already done, but in original notebook)
# random_state=42 ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions using the best estimator found by RandomizedSearchCV
# rscv.best_estimator_ is the fitted estimator (pipeline) that gave the best score
y_pred_rscv = rscv.best_estimator_.predict(X_test)
```

```python
# Verify the shapes of our training and testing sets (already done, but in original notebook)
# Should be: X_train(56000, 784), X_test(14000, 784)
X_train.shape, X_test.shape

# Evaluate the performance of our tuned model using the classification report
print(classification_report(y_test, y_pred_rscv))
```

Let's also check performance on the training set to detect potential overfitting:

```python
# Make predictions on the training set using the best estimator
y_pred_tr = rscv.best_estimator_.predict(X_train)
```

### Confusion Matrix Visualization

Let's visualize the confusion matrix to better understand our model's performance:

```python
# Import ConfusionMatrixDisplay for visualizing confusion matrices
from sklearn.metrics import ConfusionMatrixDisplay

# Visualize the confusion matrix for training predictions
# This helps to see how well the model performs on the training data and identify misclassifications
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_tr)
```

### Manual Model Tuning

Let's create a manually tuned Random Forest with specific hyperparameters to reduce overfitting:

```python
# Import required classes for creating a machine learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline that combines preprocessing and model training
# Step 1: 'scale' - Standardize features (mean=0, std=1)
# Step 2: 'model' - Random Forest classifier with manually tuned hyperparameters
pipe = Pipeline([
    ('scale', StandardScaler()),  # Preprocessing step
    ('model', RandomForestClassifier(
        n_estimators=100,       # Number of trees in the forest
        max_depth=15,           # Limit tree depth to prevent overfitting
        min_samples_split=5,    # Minimum number of samples required to split an internal node
        min_samples_leaf=3,     # Minimum number of samples required to be at a leaf node
        max_features='sqrt',    # Number of features to consider when looking for the best split (reduces overfitting in high-dimensional data)
        random_state=42,        # For reproducible results
        n_jobs=-1               # Use all available processors
    ))  # Model step
])

# Train the manually tuned pipeline on the training data
pipe.fit(X_train, y_train)
```

```python
# Make predictions on the test set using the manually tuned classifier (clf was not defined, assuming pipe is intended)
# This line seems to have a variable name mismatch (clf vs pipe)
# Assuming 'clf' was meant to be 'pipe' or a separate RandomForestClassifier instance
# For consistency with the previous cell, we'll assume 'pipe' is the intended object.
# If 'clf' was a separate model, it should have been defined and trained.
y_pred_new = pipe.predict(X_test) # Changed clf to pipe for consistency
print(classification_report(y_test, y_pred_new))
```

```python
# Train the pipeline on our training data (already done in the previous cell, but in original notebook)
# This will first scale the features, then train the Random Forest
pipe.fit(X_train, y_train)

# Display normalized confusion matrix as percentages for the manually tuned model on the test set
# normalize='true' normalizes over the true labels (rows), showing recall
# values_format=".0%" displays values as percentages with no decimal places
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_new, normalize='true', values_format=".0%")
```

## 5. Alternative Classification Approaches

### Logistic Regression

Let's try a different algorithm - Logistic Regression, which is particularly effective for this type of classification task:

```python
# Import LogisticRegression for classification
from sklearn.linear_model import LogisticRegression
# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Create and fit a StandardScaler to the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train a Logistic Regression model
# Default parameters often work well for MNIST
log_reg_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
log_reg_model.fit(X_train_scaled, y_train)
```

Let's evaluate the logistic regression model on the training set:

```python
# Make predictions on the test set using the Random Forest pipeline (this code block is misplaced)
# The pipeline will automatically apply the same scaling transformation
# This block seems to be a remnant from the previous section.
# For Logistic Regression training evaluation, we need to predict on scaled training data.
# y_pred = pipe.predict(X_test) # This line is not relevant for Logistic Regression training evaluation
```

```python
# Make predictions on the scaled training set using the Logistic Regression model
y_pred_log_tr = log_reg_model.predict(scaler.transform(X_train)) # Fixed: added scaling

# Display confusion matrix for logistic regression training predictions
# normalize='true' normalizes over the true labels (rows), showing recall
# values_format=".0%" displays values as percentages with no decimal places
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_log_tr, normalize='true', values_format=".0%")
```

Now let's evaluate on the test set:

```python
# Import evaluation metrics (already imported, but good for context)
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, confusion_matrix, classification_report

# Make predictions on the scaled test set using the Logistic Regression model
y_pred_log = log_reg_model.predict(scaler.transform(X_test)) # Fixed: added scaling

# Calculate and display various performance metrics for Logistic Regression on the test set
print("Accuracy: ", accuracy_score(y_test, y_pred_log))  # Overall accuracy
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))  # 10x10 matrix showing true vs predicted
print("Multilabel Confusion Matrix:\n", multilabel_confusion_matrix(y_test, y_pred_log))  # One matrix per class
```

```python
# Import classification_report and ConfusionMatrixDisplay (already imported, but good for context)
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Assuming y_test and y_pred_log are already defined

# Display detailed classification report with precision, recall, and F1-score for each digit
print(classification_report(y_test, y_pred_log))

# Display normalized confusion matrix as percentages for Logistic Regression on the test set
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_log, normalize='true', values_format=".0%")
```

### Multi-class Classification Strategies

#### One-vs-One (OvO) Classification

The One-vs-One strategy trains one classifier for each pair of classes. For 10 classes, this means 45 classifiers (10 choose 2):

```python
# Import OneVsOneClassifier for multi-class strategy
from sklearn.multiclass import OneVsOneClassifier
# Import LogisticRegression as the base estimator for OvO
from sklearn.linear_model import LogisticRegression

# Create One-vs-One classifier using Logistic Regression as base estimator
# n_jobs=-1: Use all available processors for training multiple classifiers
# This will create C(10, 2) = 45 binary classifiers (one for each pair of digits)
ovo_clf = OneVsOneClassifier(LogisticRegression(max_iter=1000), n_jobs=-1) # Increased max_iter for convergence
```

```python
# Import necessary libraries (already imported, but good for context)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define the model and pipeline (this block seems to be a remnant from previous sections)
# Note: 'model' here is a placeholder. Replace it with the actual model variable if named differently.
# pipe = Pipeline([
#     ('model', RandomForestClassifier())
# ])

# Define hyperparameter distributions to search (remnant)
# param_dist = {
#     "model__n_estimators": [25, 50, 100, 200, 300],
#     "model__criterion": ['gini', 'entropy', 'log_loss'],
#     "model__max_depth": [None, 0, 1, 2, 3, 5, 10]
# }

# Create RandomizedSearchCV object (remnant)
# rscv = RandomizedSearchCV(pipe, param_dist, n_jobs=-1, cv=3, verbose=True)

# Fit the One-vs-One classifier on the training data
# Note: OvO handles scaling internally for each sub-classifier if the base estimator is sensitive to scale
ovo_clf.fit(X_train, y_train)
```

Let's evaluate the OvO classifier on training data:

```python
# Make predictions on the training set using the One-vs-One classifier
y_pred_ovo_tr = ovo_clf.predict(X_train)
```

```python
# Fit the RandomizedSearchCV to find the best hyperparameters (remnant)
# This will try different combinations and select the best one using cross-validation
# rscv.fit(X_train, y_train)

# Display confusion matrix for OvO training predictions
# normalize='true' normalizes over the true labels (rows), showing recall
# values_format=".0%" displays values as percentages with no decimal places
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_ovo_tr, normalize='true', values_format=".0%")
```

```python
# Check the number of binary classifiers created by the One-vs-One strategy
# For 10 classes, this should be 45 (10 * 9 / 2)
len(ovo_clf.estimators_)
```

Now let's evaluate the OvO classifier on the test set:

```python
# Make predictions using the best estimator found by RandomizedSearchCV (remnant)
# y_pred_rscv = rscv.best_estimator_.predict(X_test)
```

```python
# Evaluate the performance of our tuned model (remnant)
# print(classification_report(y_test, y_pred_rscv))

# Make predictions on the test set using the One-vs-One classifier
y_pred_ovo = ovo_clf.predict(X_test)

# Display normalized confusion matrix as percentages for OvO on the test set
# This helps identify which digit pairs are most commonly confused
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ovo, normalize='true', values_format=".0%")
```

#### One-vs-All (OvA) Classification

The One-vs-All strategy trains one classifier per class. For 10 classes, this means 10 binary classifiers:

```python
# Import OneVsRestClassifier for multi-class strategy
from sklearn.multiclass import OneVsRestClassifier
# Import LogisticRegression as the base estimator for OvA
from sklearn.linear_model import LogisticRegression

# Create One-vs-Rest classifier using Logistic Regression as base estimator
# n_jobs=-1: Use all available processors for training multiple classifiers
# This will create 10 binary classifiers (one for each digit vs. all others)
ova_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000), n_jobs=-1) # Increased max_iter for convergence
```

```python
# Train the One-vs-Rest classifier on the training data
ova_clf.fit(X_train, y_train)

# Make predictions on training set to check for overfitting (remnant)
# y_pred_tr = rscv.best_estimator_.predict(X_train)
```

Let's evaluate the OvA classifier on training data:

```python
# Make predictions on the training set using the One-vs-Rest classifier
y_pred_ova_tr = ova_clf.predict(X_train)
```

```python
# Import and display confusion matrix for training set (already imported, but good for context)
from sklearn.metrics import ConfusionMatrixDisplay

# Visualize confusion matrix for training predictions (remnant)
# ConfusionMatrixDisplay.from_predictions(y_train, y_pred_tr)
```

```python
# Check the number of binary classifiers created by the One-vs-Rest strategy
# For 10 classes, this should be 10 (one classifier per class vs. all others)
len(ova_clf.estimators_)
```

Now let's evaluate the OvA classifier on the test set:

```python
# Create a manually tuned Random Forest classifier (remnant from previous section)
# These parameters are chosen to reduce overfitting
# clf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=15,
#     min_samples_split=5,
#     min_samples_leaf=3,
#     max_features='sqrt',
#     random_state=42,
#     n_jobs=-1
# )

# Train the manually tuned model (remnant)
# clf.fit(X_train, y_train)

# Make predictions on test set using One-vs-Rest classifier
y_pred_ova = ova_clf.predict(X_test)
```

```python
# Evaluate the manually tuned Random Forest model (remnant)
# y_pred_new = clf.predict(X_test)
# print(classification_report(y_test, y_pred_new))

# Display normalized confusion matrix as percentages for OvA on the test set
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ova, normalize='true', values_format=".0%")
```

### Error Analysis

Let's analyze the errors made by our OvA classifier to understand where it struggles:

```python
# Display normalized confusion matrix as percentages (remnant)
# This helps identify which digits are most commonly confused
# ConfusionMatrixDisplay.from_predictions(
#     y_test, y_pred_new, normalize='true', values_format=".0%")

# Create sample weights that highlight misclassified examples
# This assigns a weight of 1 to misclassified examples and 0 to correctly classified ones
sample_weight = (y_pred_ova_tr != y_train)
```

```python
# Display confusion matrix for OvA training predictions, weighted by misclassifications
# This helps to visualize which specific errors the model is making on the training data
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_ova_tr, sample_weight=sample_weight, normalize="true", values_format=".0%")
```

## 6. Model Persistence

Finally, let's save our best performing model for future use:

```python
# Import joblib for model persistence (saving and loading models)
import joblib
# Import other necessary classes (already imported, but good for context)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Save the best Random Forest model found by RandomizedSearchCV
# rscv.best_estimator_ is the entire pipeline (scaler + RandomForest)
joblib.dump(rscv.best_estimator_, 'models/random_forest_clf.joblib')

print("Model saved successfully to 'models/random_forest_clf.joblib'")
print(f"Best parameters found: {rscv.best_params_}")
print(f"Best cross-validation score: {rscv.best_score_:.4f}")

# The following lines appear to be misplaced or redundant code snippets from other sections.
# They are commented out or re-arranged to maintain the flow and avoid re-execution/re-definition.

# ovo_clf = OneVsOneClassifier(LogisticRegression(), n_jobs=-1)
# # This will create 45 binary classifiers (one for each pair of digits)
# # Create One-vs-One classifier using Logistic Regression as base estimator

# ConfusionMatrixDisplay.from_predictions(y_train, y_pred_log_tr, normalize='true', values_format=".0%")
# # Display confusion matrix for logistic regression training predictions
# joblib.dump(rscv.best_estimator_, 'random_forest_clf.joblib')
# len(ova_clf.estimators_)
# y_train, y_pred_ova_tr, normalize='true', values_format=".0%")ConfusionMatrixDisplay.from_predictions(
# # Display confusion matrix for OvA training predictions

# log_reg_model.fit(X_train_scaled, y_train)
# log_reg_model = LogisticRegression()

# # Default parameters work well for MNIST
# # Create and train Logistic Regression model
# X_train_scaled = scaler.fit_transform(X_train)
# # Should be 10 for 10 classes (one classifier per class vs. all others)
# # Check the number of binary classifiers created

# len(ovo_clf.estimators_)
# # Should be 45 for 10 classes: C(10,2) = 10!/(2!(10-2)!) = 45

# scaler = StandardScaler()

# # Create and fit scaler (Logistic Regression is sensitive to feature scales)

# # Import and set up Logistic Regression
# # Dump the best estimator of the random search cross-validation to a file

# joblib.dump(rscv.best_estimator_, 'random_forest_clf.joblib')
# ova_clf.fit(X_train, y_train)

# y_test, y_pred_log, normalize='true', values_format=".0%")ConfusionMatrixDisplay.from_predictions(
# # Display test set confusion matrix for logistic regression

# joblib.dump(rscv.best_estimator_, 'random_forest_clf.joblib')
# sample_weight = (y_pred_ova_tr != y_train)
# ovo_clf.fit(X_train, y_train)
# # Note: We use original X_train (OvO handles scaling internally for each sub-classifier)
# # Train the One-vs-Rest classifier

# # Assuming rscv, y_train, and y_pred_log_tr are defined earlier in the code

# # Dump the trained model to a file

# # This gives weight 1 to errors and weight 0 to correct predictions
# y_pred_ova_tr = ova_clf.predict(X_train)
# y_pred_ovo = ovo_clf.predict(X_test)
# # Dump the trained Random Forest classifier
# from sklearn.linear_model import LogisticRegression

# # To load the model later, use: loaded_model = joblib.load('random_forest_clf.joblib')

# joblib.dump(rscv.best_estimator_, 'random_forest_clf.joblib')
# # This includes both the preprocessing pipeline and the tuned Random Forest
# y_pred_log_tr = log_reg_model.predict(scaler.transform(X_train))  # Fixed: added scaling
# # Make predictions on test set using One-vs-One classifier

# # Create sample weights that highlight misclassified examples

# # Train the One-vs-One classifier

# # Check the number of binary classifiers created

# # Save the best model found by RandomizedSearchCV

# # Make predictions on training set using One-vs-Rest classifier
# from sklearn.preprocessing import StandardScaler

# # Make predictions on training set (need to scale the data first)

# y_test, y_pred_ova, normalize='true', values_format=".0%")

# ova_clf = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)

# # This will create 10 binary classifiers (one for each digit vs. all others)
# # Create One-vs-Rest classifier using Logistic Regression as base estimator
# joblib.dump(rscv.best_estimator_, 'random_forest_clf.joblib')
# y_pred_ova = ova_clf.predict(X_test)
# ConfusionMatrixDisplay.from_predictions(
# # Display test set confusion matrix for One-vs-Rest classifier

# ConfusionMatrixDisplay.from_predictions(y_train, y_pred_ova_tr, sample_weight=sample_weight, normalize="true", values_format=".0%")ConfusionMatrixDisplay.from_predictions(y_train, y_pred_ovo_tr, normalize='true', values_format=".0%")

# y_test, y_pred_ovo, normalize='true', values_format=".0%")y_pred_log = log_reg_model.predict(scaler.transform(X_test))  # Fixed: added scaling# Make predictions on test set using One-vs-Rest classifier

# y_pred_ovo_tr = ovo_clf.predict(X_train)
# # This helps identify which digit pairs are most commonly confused
# # Display confusion matrix for OvO training predictions

# ConfusionMatrixDisplay.from_predictions(y_tran, y_pred_ova_tr)
```

### Key Points About Model Persistence:

1.  **Complete Pipeline Saved**: We save the entire `best_estimator_` which includes both the preprocessing pipeline and the tuned Random Forest model

2.  **Joblib vs Pickle**: Joblib is preferred for scikit-learn models as it's more efficient for large NumPy arrays

3.  **Model Loading**: To load the model later, simply use:
    ```python
    loaded_model = joblib.load('models/random_forest_clf.joblib')
    predictions = loaded_model.predict(new_data)
    ```

4.  **Best Practices**:
    -   Save models with clear, descriptive filenames
    -   Include version information if models are updated frequently
    -   Store model metadata (parameters, performance metrics) alongside the model

5.  **Deployment Ready**: This saved model can be directly used in production environments for making predictions on new MNIST digit images

## Conclusion

In this notebook, we explored multiple approaches to classifying MNIST handwritten digits:

1.  **Random Forest with Pipeline**: Achieved good baseline performance with automated preprocessing
2.  **Hyperparameter Tuning**: Improved performance using RandomizedSearchCV
3.  **Manual Tuning**: Created a more refined model to reduce overfitting
4.  **Logistic Regression**: Demonstrated a simpler, faster alternative approach
5.  **Multi-class Strategies**: Compared One-vs-One and One-vs-All approaches

### Key Takeaways:

-   Feature scaling is crucial for algorithms like Logistic Regression
-   Random Forest generally performs well on MNIST with minimal tuning
-   One-vs-One and One-vs-All strategies can be effective for multi-class problems
-   Confusion matrices help identify which digits are commonly misclassified
-   Proper evaluation on both training and test sets helps detect overfitting

### Next Steps:

-   Experiment with ensemble methods combining multiple models
-   Explore data augmentation techniques to improve robustness
-   Analyze misclassified examples to understand model limitations
