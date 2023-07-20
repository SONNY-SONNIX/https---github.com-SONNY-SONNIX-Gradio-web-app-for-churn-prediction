
##Data Handling
import gradio as gr
import pandas as pd 
import numpy as np 

##Visualization Libraries 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
import random
import plotly.offline as offline
offline.init_notebook_mode(connected=True) # Configure Plotly to run offline

##model evaluation:
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

# Other packages
import os
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
import joblib


#Z:\GitHub Space\Gradio-web-app-for-churn-prediction\models

absolute_path =os.path.dirname(r"/GitHub Space/Gradio-web-app-for-churn-prediction/models/LR.plk")
relative_path ="models/LR.plk"
full_path =os.path.join(absolute_path,relative_path)

#path = r"GitHub Space\Gradio-web-app-for-churn-prediction\models\LR.plk"
block= gr.Blocks(theme= "freddyaboulton/dracula_revamped")
model = joblib.load(relative_path)
#model = joblib.load("./Gradio-web-app-for-churn-prediction/models/LR.plk")

def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"



def predict_churn(SeniorCitizen, Partner, Dependents, tenure, InternetService,
                  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                  PaymentMethod, MonthlyCharges, TotalCharges):
    input_data = [
        SeniorCitizen, Partner, Dependents, tenure, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges
    ]

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    pred = model.predict(input_df)
    output = classify(pred[0])

    if output == "Customer will not Churn":
        return [(0, output)]
    else:
        return [(1, output)]


    with block:
        gr.Markdown(""" # Welcome to My Customer Churn Prediction App""")
        input=[gr.inputs.Slider(minimum=0, maximum= 1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
            gr.inputs.Radio(["Yes", "No"], label="Partner: Do You Have a Partner?"),
            gr.inputs.Radio(["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
            gr.inputs.Number(label="tenure: How Long Have You Been with Vodafone in Months?"),
            gr.inputs.Radio(["DSL", "Fiber optic", "No"], label="InternetService"),
    ]

output = gr.outputs.HighlightedText(color_map={
    "Customer will not Churn": "green",
    "Customer will churn": "red"
})

predict_btn= gr.Button("Predict")
     
predict_btn.click(fn= predict_churn, inputs= input, outputs=output)


# iface = gr.Interface(title= "Customer Churn Prediction For Vodafone PLC",
#     fn=predict_churn,
#     inputs=[
#         gr.inputs.Slider(minimum=0, maximum= 1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
#         gr.inputs.Dropdown(["Yes", "No"], label="Partner: Do You Have a Partner?"),
#         gr.inputs.Dropdown(["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
#         gr.inputs.Number(label="tenure: How Long Have You Been with Vodafone in Months?"),
#         gr.inputs.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="TechSupport"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV"),
#         gr.inputs.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies"),
#         gr.inputs.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
#         gr.inputs.Dropdown(["Yes", "No"], label="PaperlessBilling"),
#         gr.inputs.Dropdown([
#             "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
#         ], label="PaymentMethod"),
#         gr.inputs.Number(label="MonthlyCharges"),
#         gr.inputs.Number(label="TotalCharges")
#     ],
#     outputs=output,  theme="freddyaboulton/dracula_revamped"
# )

# iface.launch(share= True )

