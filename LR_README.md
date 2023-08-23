
### Linear Regression in Python - A Step-by-Step Tutorial

This tutorial provides a step-by-step guide to building a linear regression model in Python using the popular scikit-learn library. Linear regression is a supervised learning algorithm that is used to predict a continuous target variable based on one or more predictor variables. It is a widely used technique for regression and predictive modeling.

### Step 1: Import the Necessary Libraries

We start by importing the necessary libraries. In this case, we need the `numpy`, `pandas`, `matplotlib`, and `sklearn` libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

### Step 2: Load the Dataset

Next, we load the dataset that we will be using to train our linear regression model. In this example, we will be using the Boston housing dataset, which contains information on housing prices in the Boston area.

```python
data = pd.read_csv("HousingData.csv")
```

### Step 3: Select Features and Target

We then need to select the features (independent variables) and target (dependent variable) that we will be using in our model. In this case, we will be using the average number of rooms per house as the feature and the median housing price as the target.

```python
X = data[["RM"]]  # Feature: average number of rooms
y = data["MEDV"]  # Target: housing prices
```

## Step 4: ## 
`Plot/Visualize` the data to get a better understanding of the data.
```rust
# Create a scatter plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.scatter(X, y, color="blue", alpha=0.5)  # Create the scatter plot
plt.title("Scatter Plot: Average Number of Rooms vs Housing Price")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Housing Price")
plt.grid(True)
plt.show()
```

### Step 4: Split the Data into Training and Testing Sets

We then split the data into training and testing sets. This is important because it allows us to evaluate the performance of our model on unseen data. We will use 80% of the data for training and 20% for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 5: Create and Fit the Linear Regression Model

We can now create and fit the linear regression model using the `LinearRegression` class from the `sklearn` library.

```python
# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

```

### Step 6: Plot the data and Regression Line
you can now plot the data and regression line to visualize the outcome using `Matplotlib` library
```rust
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=3, label="Regression Line")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Housing Price")
plt.title("Linear Regression: Boston Housing Prices")
plt.legend()
plt.show()
```
