import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Obesity Prediction", layout="centered")

# Custom CSS for background styling
st.markdown("""
    <style>
        body {
            background-color: #E0F7FA;
            background-image: url("https://via.placeholder.com/1500x1000.png"); /* Replace with your background image URL */
            background-size: cover;
        }
        .prediction-form, .result-table {
            background: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 10px;
            max-width: 700px;
            margin: auto;
        }
        .form-title {
            font-size: 36px;
            text-align: center;
            font-weight: bold;
        }
        .predict-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open("Source/Prototype/models/obesity_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Prediction function using the model
def predict_obesity(input_data):
    input_df = pd.DataFrame([input_data])  # Convert input data to DataFrame
    prediction = model.predict(input_df)   # Make a prediction
    return prediction[0]

# Map function for CAEC and CALC inputs
def map_caec_calc(value):
    mapping = {'no': 3, 'Sometimes': 0, 'Frequently': 1, 'Always': 2}
    return mapping.get(value, 3)  # Default to 'no' if the value is unexpected

# Prediction form function
def show_prediction_form():
    st.markdown("<div class='form-title'>OBESITY PREDICTION</div>", unsafe_allow_html=True)
    
    with st.form(key="prediction_form", clear_on_submit=False):
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, step=0.01)
        weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=0.1)
        
        family_history = st.selectbox("Family with Overweight", ["yes", "no"])
        FAVC = st.selectbox("Do you eat high caloric food frequently? (FAVC)", ["yes", "no"])
        FCVC = st.number_input("How often do you eat vegetables? (FCVC)", min_value=1.0, max_value=3.0, step=1.0)
        NCP = st.number_input("Number of main meals (NCP)", min_value=1.0, max_value=4.0, step=1.0)
        CAEC = st.selectbox("Consumption of food between meals? (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
        SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])
        CH2O = st.number_input("Water intake (liters per day) (CH2O)", min_value=1.0, max_value=3.0, step=1.0)
        SCC = st.selectbox("Do you monitor calorie intake? (SCC)", ["yes", "no"])
        FAF = st.number_input("Physical activity frequency (FAF)", min_value=0.0, max_value=3.0, step=1.0)
        TUE = st.number_input("Time using technology devices (hours) (TUE)", min_value=0.0, max_value=2.0, step=1.0)
        CALC = st.selectbox("Alcohol consumption frequency (CALC)", ["no", "Sometimes", "Frequently", "Always"])

        # Submit button
        submitted = st.form_submit_button("Predict")
        if submitted:
            # Prepare data for prediction with correct encoding
            input_data = {
                "Gender": 1 if gender == "Male" else 0,
                "Age": age,
                "Height": height,
                "Weight": weight,
                "family_history_with_overweight": 1 if family_history == "yes" else 0,
                "FAVC": 1 if FAVC == "yes" else 0,
                "FCVC": FCVC,
                "NCP": NCP,
                "CAEC": map_caec_calc(CAEC),
                "SMOKE": 1 if SMOKE == "yes" else 0,
                "CH2O": CH2O,
                "SCC": 1 if SCC == "yes" else 0,
                "FAF": FAF,
                "TUE": TUE,
                "CALC": map_caec_calc(CALC)
            }

            # Get prediction
            prediction = predict_obesity(input_data)

            # Show results
            show_prediction_results(input_data, prediction)

# Prediction results function
def show_prediction_results(result_data, prediction):
    st.markdown("<div class='form-title'>PREDICTION RESULT</div>", unsafe_allow_html=True)
    
    # Convert the result_data to DataFrame for displaying in table format
    result_df = pd.DataFrame.from_dict(result_data, orient='index', columns=["Value"])
    
    # Display result table
    st.markdown("<div class='result-table'>", unsafe_allow_html=True)
    st.table(result_df)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display prediction result below the table with a better format
    st.metric(label="Obesity Prediction", value=f"{prediction}", delta="")

    
    # # Interactive Histogram for Age and Weight distribution
    # hist_fig = px.histogram(result_df, x="Value", title="Age and Weight Distribution", labels={'Value': 'Features'})
    # st.plotly_chart(hist_fig)

    # Line chart for tracking prediction changes (just an example)
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Value'], mode='lines+markers', name='Input Features'))
    line_fig.update_layout(title='Feature Progression for Obesity Prediction', xaxis_title='Features', yaxis_title='Values')
    st.plotly_chart(line_fig)

# Main app logic
if "show_results" not in st.session_state:
    st.session_state["show_results"] = False

if not st.session_state["show_results"]:
    show_prediction_form()
else:
    show_prediction_results()
