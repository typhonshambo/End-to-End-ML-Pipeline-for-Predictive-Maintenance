# File: generate_data.py

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(0)

# Generate example data
num_samples = 1000
timestamps = pd.date_range(start='2022-01-01', periods=num_samples, freq='H')

# Generate numeric features
numeric_feature_mean = 100
numeric_feature_std = 20
numeric_feature = np.random.normal(numeric_feature_mean, numeric_feature_std, num_samples)

# Generate categorical features
categories = ['A', 'B', 'C']
categorical_feature = np.random.choice(categories, size=num_samples)

# Create DataFrame
data = pd.DataFrame({'timestamp': timestamps,
                     'numeric_feature': numeric_feature,
                     'categorical_feature': categorical_feature})

# Save data to CSV file
data.to_csv('data/raw/simulated_data.csv', index=False)
