# Sales Prediction Using Python
# Oasis Infobyte - Data Science Task 5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/Advertising.csv")

print("=" * 50)
print("SALES PREDICTION USING MACHINE LEARNING")
print("=" * 50)

print("\nDataset Loaded Successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Exploratory Data Analysis

# Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["sales"], kde=True, bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.savefig("output/sales_distribution.png")
plt.show()

# Advertising Budget vs Sales
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.scatterplot(data=df, x="TV", y="sales", ax=axes[0])
axes[0].set_title("TV Advertising vs Sales")

sns.scatterplot(data=df, x="radio", y="sales", ax=axes[1])
axes[1].set_title("Radio Advertising vs Sales")

sns.scatterplot(data=df, x="newspaper", y="sales", ax=axes[2])
axes[2].set_title("Newspaper Advertising vs Sales")

plt.tight_layout()
plt.savefig("output/advertising_vs_sales.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.show()

print("Exploratory Data Analysis Completed!")
# Model Training and Evaluation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features and Target
X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

    print(f"\nModel: {name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Save Model Comparison
results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2 Score"]
)
results_df.to_csv("output/model_comparison.csv", index=False)

print("\nModel Training Completed!")

# Actual vs Predicted Sales

best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(), "Model"
]

best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales - {best_model_name}")
plt.savefig("output/actual_vs_predicted.png")
plt.show()

# Save Prediction Results
prediction_df = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})
prediction_df.to_csv("output/prediction_results.csv", index=False)

# Feature Importance
rf_model = models["Random Forest"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("output/feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.savefig("output/feature_importance.png")
plt.show()

# Predict Sales for New Advertising Budget
new_data = pd.DataFrame({
    "TV": [200],
    "radio": [40],
    "newspaper": [30]
})

new_prediction = best_model.predict(new_data)

print("\nBest Model:", best_model_name)
print("New Advertising Budget:")
print(new_data)
print(f"Predicted Sales: {new_prediction[0]:.2f}")

print("\nSales Prediction Project Completed!")