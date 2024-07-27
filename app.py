import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
model = pickle.load(open('Rf_Model.sav', 'rb'))

# Define the categorical features and their possible values
categorical_features = {
    'gender': ['Female', 'Male'],
    'Partner': ['No', 'Yes'],
    'Dependents': ['No', 'Yes'],
    'PhoneService': ['No', 'Yes'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No internet service', 'No', 'Yes'],
    'OnlineBackup': ['No internet service', 'No', 'Yes'],
    'DeviceProtection': ['No internet service', 'No', 'Yes'],
    'TechSupport': ['No internet service', 'No', 'Yes'],
    'StreamingTV': ['No internet service', 'No', 'Yes'],
    'StreamingMovies': ['No internet service', 'No', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
}

# Define encoder dictionary
encoder_dict = {feature: LabelEncoder().fit(choices) for feature, choices in categorical_features.items()}

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")

st.title('Customer Churn Prediction')

# Input fields
st.sidebar.header('Enter Customer Details')

input_data = {}

# Creating input boxes for each feature
input_data['customerID'] = st.sidebar.text_input('Customer ID')
input_data['gender'] = st.sidebar.selectbox('Gender', options=categorical_features['gender'])
input_data['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', options=[0, 1])
input_data['Partner'] = st.sidebar.selectbox('Partner', options=categorical_features['Partner'])
input_data['Dependents'] = st.sidebar.selectbox('Dependents', options=categorical_features['Dependents'])
input_data['tenure'] = st.sidebar.number_input('Tenure', value=0, step=1)
input_data['PhoneService'] = st.sidebar.selectbox('Phone Service', options=categorical_features['PhoneService'])
input_data['MultipleLines'] = st.sidebar.selectbox('Multiple Lines', options=categorical_features['MultipleLines'])
input_data['InternetService'] = st.sidebar.selectbox('Internet Service', options=categorical_features['InternetService'])
input_data['OnlineSecurity'] = st.sidebar.selectbox('Online Security', options=categorical_features['OnlineSecurity'])
input_data['OnlineBackup'] = st.sidebar.selectbox('Online Backup', options=categorical_features['OnlineBackup'])
input_data['DeviceProtection'] = st.sidebar.selectbox('Device Protection', options=categorical_features['DeviceProtection'])
input_data['TechSupport'] = st.sidebar.selectbox('Tech Support', options=categorical_features['TechSupport'])
input_data['StreamingTV'] = st.sidebar.selectbox('Streaming TV', options=categorical_features['StreamingTV'])
input_data['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', options=categorical_features['StreamingMovies'])
input_data['Contract'] = st.sidebar.selectbox('Contract', options=categorical_features['Contract'])
input_data['PaperlessBilling'] = st.sidebar.selectbox('Paperless Billing', options=categorical_features['PaperlessBilling'])
input_data['PaymentMethod'] = st.sidebar.selectbox('Payment Method', options=categorical_features['PaymentMethod'])
input_data['MonthlyCharges'] = st.sidebar.number_input('Monthly Charges', value=0.0, step=0.1)
input_data['TotalCharges'] = st.sidebar.number_input('Total Charges', value=0.0, step=0.1)

# Preprocess input data
input_df = pd.DataFrame([input_data])

# Encode categorical features
for feature, encoder in encoder_dict.items():
    input_df[feature] = encoder.transform(input_df[feature])

# Drop 'customerID' as it's not a feature used for prediction
input_df = input_df.drop(columns=['customerID'])

# Ensure the same number of features are used as in training
selected_columns = ['Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']
input_df = input_df[selected_columns]

# Predict button
if st.button('Predict'):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
    except Exception as e:
        st.write("Error during prediction:", str(e))
