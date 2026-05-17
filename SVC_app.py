import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Title
st.title("Credit Card Prediction using SVC")

# Load dataset
df = pd.read_csv("UniversalBank (1).csv")

# Drop unnecessary columns
df.drop(['ID', 'ZIP Code'], axis=1, inplace=True)

# Features and target
X = df.drop('CreditCard', axis=1)
y = df['CreditCard']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = SVC(kernel='linear')
model.fit(x_train, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

Age = st.sidebar.number_input("Age", 18, 100, 30)
Experience = st.sidebar.number_input("Experience", 0, 50, 5)
Income = st.sidebar.number_input("Income", 0, 300, 50)
Family = st.sidebar.number_input("Family", 1, 10, 2)
CCAvg = st.sidebar.number_input("CCAvg", 0.0, 20.0, 2.0)
Education = st.sidebar.number_input("Education", 1, 3, 1)
Mortgage = st.sidebar.number_input("Mortgage", 0, 1000, 0)
SecuritiesAccount = st.sidebar.number_input("Securities Account", 0, 1, 0)
CDAccount = st.sidebar.number_input("CD Account", 0, 1, 0)
Online = st.sidebar.number_input("Online", 0, 1, 1)
CreditCard = st.sidebar.number_input("CreditCard", 0, 1, 1)

# Create input dataframe
input_data = pd.DataFrame([[
    Age,
    Experience,
    Income,
    Family,
    CCAvg,
    Education,
    Mortgage,
    SecuritiesAccount,
    CDAccount,
    Online,
    CreditCard
]], columns=X.columns)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Customer is likely to use Credit Card")
    else:
        st.error("Customer is NOT likely to use Credit Card")
