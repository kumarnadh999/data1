import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Define features and target variable
X = df[['Year', 'Kilometers_Driven', 'Engine', 'Power', 'Seats']]
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

# Streamlit app
st.title('Car Price Prediction')
st.write('This is a simple linear regression model to predict car prices.')

# Display the dataset
st.subheader('Dataset')
st.write(df)

# Display model evaluation
st.subheader('Model Evaluation')
st.write(f'Mean Squared Error: {mse}')

# User input for prediction
st.subheader('Make a Prediction')
year = st.number_input('Year', min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=int(df['Year'].mean()))
kilometers_driven = st.number_input('Kilometers Driven', min_value=int(df['Kilometers_Driven'].min()), max_value=int(df['Kilometers_Driven'].max()), value=int(df['Kilometers_Driven'].mean()))
engine = st.number_input('Engine (CC)', min_value=int(df['Engine'].str.replace(' CC', '').astype(float).min()), max_value=int(df['Engine'].str.replace(' CC', '').astype(float).max()), value=int(df['Engine'].str.replace(' CC', '').astype(float).mean()))
power = st.number_input('Power (bhp)', min_value=int(df['Power'].str.replace(' bhp', '').astype(float).min()), max_value=int(df['Power'].str.replace(' bhp', '').astype(float).max()), value=int(df['Power'].str.replace(' bhp', '').astype(float).mean()))
seats = st.number_input('Seats', min_value=int(df['Seats'].min()), max_value=int(df['Seats'].max()), value=int(df['Seats'].mean()))

# Make prediction
if st.button('Predict'):
    prediction = model.predict([[year, kilometers_driven, engine, power, seats]])
    st.write(f'Predicted Price: {prediction[0]}')
