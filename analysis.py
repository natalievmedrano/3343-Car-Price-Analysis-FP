import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("visuals", exist_ok=True)

df = pd.read_csv("data/used_cars.csv")

df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["milage"] = (
    df["milage"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" mi.", "", regex=False)
    .astype(float)
)

current_year = 2026
df["car_age"] = current_year - df["model_year"]

df = df.dropna(subset=["fuel_type"])

print("Shape after cleaning:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df[["price", "milage", "model_year", "car_age"]].describe())

# Mileage vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["milage"], df["price"], alpha=0.4)
plt.xlabel("Mileage")
plt.ylabel("Price (USD)")
plt.title("Mileage vs Used Car Price")
plt.tight_layout()
plt.savefig("visuals/mileage_vs_price.png")
plt.show()

# Car Age vs Price
plt.figure(figsize=(8, 5))
plt.scatter(df["car_age"], df["price"], alpha=0.4)
plt.xlabel("Car Age")
plt.ylabel("Price (USD)")
plt.title("Car Age vs Used Car Price")
plt.tight_layout()
plt.savefig("visuals/age_vs_price.png")
plt.show()

# Brand vs Price, excluding extreme luxury outliers
luxury_outliers = ["Bugatti", "Rolls-Royce", "Lamborghini"]
brand_data = df[~df["brand"].isin(luxury_outliers)]

brand_avg = (
    brand_data.groupby("brand")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(10, 5))
brand_avg.plot(kind="bar")
plt.xlabel("Brand")
plt.ylabel("Average Price (USD)")
plt.title("Top 15 Brands by Average Used Car Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/brand_vs_price.png")
plt.show()

# Fuel Type vs Price, excluding unclear fuel labels
fuel_data = df[~df["fuel_type"].isin(["–", "not supported"])]

fuel_avg = (
    fuel_data.groupby("fuel_type")["price"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
fuel_avg.plot(kind="bar")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price (USD)")
plt.title("Average Used Car Price by Fuel Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/fuel_vs_price.png")
plt.show()

# Correlation
correlation = df[["price", "milage", "model_year", "car_age"]].corr()
print("\nCorrelation matrix:")
print(correlation)