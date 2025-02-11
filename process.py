import pandas as pd

# Load the dataset
df = pd.read_csv('used_cars_data.csv')

# Display basic information
print(df.info())

# Handle missing values, encode categorical variables, etc.
df = df.dropna()  # Example: Drop rows with missing values

# Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
