import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
file_path = r'C:\Users\Sindhu\Downloads\archive (5)\Mall_Customers.csv'
data = pd.read_csv(file_path)

# Select relevant columns for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# K-means clustering (k=5 for demonstration; adjust k after elbow method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Output clustered results (first 10 rows)
print(data.head(10))
