import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("HousingData.csv")
# Print column names
print(data.columns)


# Select features and target
X = data[["RM"]]  # Feature: average number of rooms
y = data["MEDV"]  # Target: housing prices
# Create a scatter plot
plt.figure(figsize=(10, 6))  # Set the figure size
plt.scatter(X, y, color="blue", alpha=0.5)  # Create the scatter plot
plt.title("Scatter Plot: Average Number of Rooms vs Housing Price")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Housing Price")
plt.grid(True)
plt.show()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Plot the data and regression line
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=3, label="Regression Line")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Housing Price")
plt.title("Linear Regression: Boston Housing Prices")
plt.legend()
plt.show()

print("Mean Squared Error:", mse)
# Ask the user for input
user_input = float(input("Enter the average number of rooms: "))
user_input = np.array([[user_input]])  # Reshape the input for prediction

# Use the trained model to make a prediction
predicted_price = model.predict(user_input)

print(f"Predicted Housing Price: ${predicted_price[0]:.2f}")

from sklearn.metrics import r2_score
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# Print the R-squared value
print("R-squared:", r2)


