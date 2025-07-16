import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Hours': [1.5, 2.5, 3, 3.5, 4.5, 5, 6.1, 6.5, 7.8, 8.2],
    'Marks': [35, 40, 45, 50, 55, 60, 68, 70, 80, 85]
}
df = pd.DataFrame(data)

# Train the model
X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("📊 Student Performance Predictor")
st.write("Enter study hours to predict marks")

hours = st.slider("Study Hours", 1.0, 10.0, 5.0)
predicted_marks = model.predict([[hours]])[0]

st.success(f"If you study **{hours:.1f} hours**, you may score around **{predicted_marks:.2f} marks**.")
