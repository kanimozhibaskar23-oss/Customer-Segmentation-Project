import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 1. Project Start
print("Customer Segmentation Project")
print("Libraries imported successfully!")


# 2. Load Dataset
df = pd.read_csv("customer_dataset.csv")

print("\nDataset loaded successfully!")

print("\nFirst 5 customers:")
print(df.head())


# 3. Dataset Information
print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# 4. Select Features
features = df[
    ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency"]
]


# 5. Feature Scaling
scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

print("\nFeatures scaled successfully!")


# 6. K-Means Clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_features)

print("\nCustomer clusters created successfully!")

print(df[["CustomerID", "Cluster"]])


# 7. Cluster Count
print("\nNumber of customers in each cluster:")

print(df["Cluster"].value_counts())


# 8. Cluster Analysis
cluster_summary = df.groupby("Cluster")[
    [
        "Age",
        "AnnualIncome",
        "SpendingScore",
        "PurchaseFrequency"
    ]
].mean()

print("\nCluster Analysis:")

print(cluster_summary)


# 9. Save Segmented Data
df.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nCustomer segmentation results saved successfully!")


# 10. Create Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"],
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation using K-Means")

plt.grid(True)

# Save graph
plt.savefig(
    "customer_segments.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCustomer segmentation graph saved successfully!")

print("\nPROJECT COMPLETED SUCCESSFULLY!")