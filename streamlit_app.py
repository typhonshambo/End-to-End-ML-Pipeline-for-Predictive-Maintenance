import streamlit as st
import pandas as pd
import numpy as np
from pipelines.prediction_pipeline import prediction_pipeline
from pipelines.config import DataPaths
import shutil
# Title
st.title('Predictive Maintenance Model')
st.write('A predictive maintenance ML model that predicts whether a device will fail or not based on the past metrics dataset, check the model monitoring dashboard for more details. You can download and upload the sample dataset to make predictions')

sample_dataset_url = 'https://github.com/typhonshambo/End-to-End-ML-Pipeline-for-Predictive-Maintenance/blob/main/data/raw/preview.csv'
dagshub_uri = DataPaths.mlflow_uri

if st.button('Download Sample Dataset'):
    st.markdown(f"[Download Sample Dataset]({sample_dataset_url})", unsafe_allow_html=True)

if st.button('Go to Model Monitoring Dashboard'):
    st.markdown(f"[MLFlow Dashboard]({dagshub_uri})", unsafe_allow_html=True)

# Option to select input method
input_method = st.radio("Choose input method:", ('Upload CSV', 'Manual Input'))

if input_method == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data from CSV:")
        st.write(data)
        data.to_csv(DataPaths.steamlit_uploaded_data, index=False)
        if st.button('Predict'):
            try:
                result = prediction_pipeline(DataPaths.steamlit_uploaded_data)
                st.write(result)
            except Exception as e:
                st.write("```ERROR : Please check the uploaded data. It should have the following columns: ```")
                input_data = pd.DataFrame({
                    'date': ['1/1/2015'],  
                    'device': ['S1F01085'],  
                    'metric1': ['215630672'],  
                    'metric2': ['55'],  
                    'metric3': ['0'],  
                    'metric4': ['52'],  
                    'metric5': ['6'],  
                    'metric6': ['407438'],  
                    'metric7': ['0'],  
                    'metric8': ['0'],  
                    'metric9': ['7']  
                }, index=[0])  
                st.write(input_data)
                st.write("You can download sample data from the 'Download Sample Dataset' button above and upload.")

else:
    date = st.date_input('Date')
    device = st.text_input('Device')
    metric1 = st.number_input('Metric 1')
    metric2 = st.number_input('Metric 2')
    metric3 = st.number_input('Metric 3')
    metric4 = st.number_input('Metric 4')
    metric5 = st.number_input('Metric 5')
    metric6 = st.number_input('Metric 6')
    metric7 = st.number_input('Metric 7')
    metric8 = st.number_input('Metric 8')
    metric9 = st.number_input('Metric 9')

    input_data = pd.DataFrame({
        'date': [date],
        'device': [device],
        'metric1': [metric1],
        'metric2': [metric2],
        'metric3': [metric3],
        'metric4': [metric4],
        'metric5': [metric5],
        'metric6': [metric6],
        'metric7': [metric7],
        'metric8': [metric8],
        'metric9': [metric9]
    })

    st.write("Manual Input Data:")
    st.write(input_data)
    input_data.to_csv(DataPaths.steamlit_uploaded_data, index=False)
    if st.button('Predict'):
        try:
            result = prediction_pipeline(DataPaths.steamlit_uploaded_data)
            st.write(result)
        except Exception as e:
            st.write("```ERROR in prediction```",)
            st.write("```your data should look like this: ```",)
            input_data = pd.DataFrame({
                'date': ['1/1/2015'],  
                'device': ['S1F01085'],  
                'metric1': ['215630672'],  
                'metric2': ['55'],  
                'metric3': ['0'],  
                'metric4': ['52'],  
                'metric5': ['6'],  
                'metric6': ['407438'],  
                'metric7': ['0'],  
                'metric8': ['0'],  
                'metric9': ['7']  
            }, index=[0])  
            st.write(input_data)
            st.write("You can download sample data from the 'Download Sample Dataset' button above and upload.")
            