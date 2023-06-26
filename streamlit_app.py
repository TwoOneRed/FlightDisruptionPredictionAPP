import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from powerbiclient import Report

st.title('Flight Disruption Prediction Applications')
st.text('1191100292  Leong Yi Hong')

scope = [ 'https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("key.json",scope)
client = gspread.authorize(credentials)


uploaded_file = st.file_uploader("New Dataset", type="csv")

if uploaded_file:
    # Do something with the uploaded file
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(5))
    sheet = client.open("FYP2Data_PowerBI").sheet1
    # Clear the existing data from the sheet
    sheet.clear()

    #Convert To String
    newdata = df.astype(str)

    # Update the Google Sheets file with the modified DataFrame
    st.write(sheet.update([newdata.columns.values.tolist()] + newdata.values.tolist()))
    

def get_markdown_content():
    return """<iframe title="PB" width="900" height="550" src="https://app.powerbi.com/view?r=eyJrIjoiMmU3ZjFmODItMDE1My00ZDE4LWJhNmQtOTFiYmM1ODAxYWU4IiwidCI6IjdlMGI1ZmNmLTEyYzQtNGVmZi05NmI2LTQ2NjRmMjVkYzdkYSIsImMiOjEwfQ%3D%3D&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>"""

# Create a Streamlit app
st.title("Power BI Dashboard")

# Add a refresh button
refresh_button = st.button("Refresh")

# Display the report
if refresh_button:
    markdown_content = get_markdown_content()
    st.markdown(markdown_content, unsafe_allow_html=True)
else:
    # Load the initial content
    st.markdown(get_markdown_content(), unsafe_allow_html=True)

