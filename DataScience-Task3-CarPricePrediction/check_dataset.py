
import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/car_data.csv")

print("DATASET LOADED SUCCESSFULLY")
print("=" * 45)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)
