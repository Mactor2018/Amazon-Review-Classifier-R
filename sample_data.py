"""Create a reproducible 10% sample of amazon_reviews.csv (-> data_small.csv).

Usage: python sample_data.py [input.csv] [output.csv]
Use the result with: REVIEWS_CSV=data_small.csv Rscript data_preprocessing.R
"""
import sys

import pandas as pd

input_file = sys.argv[1] if len(sys.argv) > 1 else "amazon_reviews.csv"
output_file = sys.argv[2] if len(sys.argv) > 2 else "data_small.csv"

df = pd.read_csv(input_file)
sample = df.sample(frac=0.1, random_state=42)
sample.to_csv(output_file, index=False)
print(f"{input_file}: {len(df)} rows -> {output_file}: {len(sample)} rows")
