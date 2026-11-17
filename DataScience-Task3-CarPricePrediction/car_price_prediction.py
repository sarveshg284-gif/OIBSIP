
# CAR PRICE PREDICTION USING MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("dataset/car_data.csv")

print("Dataset Loaded Successfully!")
print("Original Dataset Shape:", df.shape)

# 2. Data Cleaning
df = df.drop_duplicates()

# 3. Feature Engineering
df["Car_Age"] = 2026 - df["Year"]

# 4. Select Features and Target
X = df[[
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Car_Age"
]]

y = df["Selling_Price"]

# 5. Encode categorical columns
categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

best_model = None
best_score = -float("inf")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel:", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_name = name
        best_predictions = predictions

# 8. Best Model Results
print("\nBest Model:", best_name)
print("Best R2 Score:", round(best_score, 2))

# 9. Actual vs Predicted Graph
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.savefig("output/actual_vs_predicted.png")
plt.show()

# 10. Selling Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Selling Price Distribution")
plt.savefig("output/price_distribution.png")
plt.show()

print("\nProject Completed Successfully!")

# 11. Feature Importance / Coefficients
if best_name == "Linear Regression":
    model = best_model.named_steps["model"]
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.coef_
    })

    importance["Absolute_Importance"] = importance["Importance"].abs()
    importance = importance.sort_values(
        by="Absolute_Importance", ascending=False
    )

else:
    model = best_model.named_steps["model"]
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance[["Feature", "Importance"]])

# Save Feature Importance
importance.to_csv("output/feature_importance.csv", index=False)

# Feature Importance Graph
plt.figure(figsize=(10, 6))
sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.show()

# 12. Save Prediction Results
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": best_predictions
})

results.to_csv("output/prediction_results.csv", index=False)

print("\nFeature importance and prediction results saved!")

