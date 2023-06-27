
##Data Handling
import gradio as gr
import pandas as pd 
import numpy as np 

##Visualization Libraries 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
%matplotlib inline
import random
import plotly.offline as offline
offline.init_notebook_mode(connected=True) # Configure Plotly to run offline

# Other packages
import os
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import joblib

iface = gr.Interface(title= "Customer Churn Prediction For Vodafone PLC",
    fn=predict_churn,
    inputs=[
        gr.inputs.Slider(minimum=0, maximum= 1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
        gr.inputs.Dropdown(["Yes", "No"], label="Partner: Do You Have a Partner?"),
        gr.inputs.Dropdown(["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
        gr.inputs.Number(label="tenure: How Long Have You Been with Vodafone in Months?"),
        gr.inputs.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="TechSupport"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV"),
        gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies"),
        gr.inputs.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.inputs.Dropdown(["Yes", "No"], label="PaperlessBilling"),
        gr.inputs.Dropdown([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="PaymentMethod"),
        gr.inputs.Number(label="MonthlyCharges"),
        gr.inputs.Number(label="TotalCharges")
    ],
    outputs=output,  theme="freddyaboulton/dracula_revamped"
)

iface.launch(share= True )