import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file
input_file = r'D:\Coding\R\Group\Amazon-Review-Classifier\amazon_reviews.csv'
output_file = r'D:\Coding\R\Group\Amazon-Review-Classifier\data_new.csv'

print(f"Reading data from {input_file}...")
df = pd.read_csv(input_file)

print(f"Original data shape: {df.shape}")
print(f"Total rows: {len(df)}")

# Sample 10% of the data uniformly
sample_size = int(len(df) * 0.1)
df_sampled = df.sample(n=sample_size, random_state=42)

print(f"Sampled data shape: {df_sampled.shape}")
print(f"Sampled rows: {len(df_sampled)}")

# Save to CSV
df_sampled.to_csv(output_file, index=False)
print(f"Sampled data saved to {output_file}")


