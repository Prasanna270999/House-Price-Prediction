import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()

# Display first 5 feature rows
print("Features:")
print(housing.data[:5])

# Display first 5 target values
print("\nHouse Prices:")
print(housing.target[:5])

# Convert to DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add target column
df["Price"] = housing.target

print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation Matrix

plt.figure(figsize=(10, 8))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# Features and Target
X = housing.data
y = housing.target

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")
# Predictions
predictions = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(predictions[:10])

print("\nActual Prices:")
print(y_test[:10])
# Model Evaluation

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)
# Actual vs Predicted Plot

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()
#scatter plot
plt.figure(figsize=(10,6))

plt.plot(y_test[:100], label="Actual Prices")
plt.plot(predictions[:100], label="Predicted Prices")

plt.title("Actual vs Predicted Prices")
plt.xlabel("Sample Index")
plt.ylabel("House Price")

plt.legend()
plt.show()