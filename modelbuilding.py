import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Define features and target variable
X = df[['Year', 'Kilometers_Driven', 'Engine', 'Power', 'Seats']]  # Replace with actual feature names
y = df['Price']

# Convert categorical features to numerical values if necessary
X['Engine'] = X['Engine'].str.replace(' CC', '').astype(float)
X['Power'] = X['Power'].str.replace(' bhp', '').astype(float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
