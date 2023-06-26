import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pickle
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

st.title('Flight Disruption Prediction Applications')
st.text('1191100292  Leong Yi Hong')

scope = [ 'https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("key.json",scope)
client = gspread.authorize(credentials)

st.title("Upload Dataset")
uploaded_file = st.file_uploader("New Dataset", type="csv")

df = pd.DataFrame()

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

# Create a Streamlit app
st.title("Power BI Dashboard")

# Display the report
st.markdown("""<iframe title="PB" width="900" height="550" src="https://app.powerbi.com/view?r=eyJrIjoiMmU3ZjFmODItMDE1My00ZDE4LWJhNmQtOTFiYmM1ODAxYWU4IiwidCI6IjdlMGI1ZmNmLTEyYzQtNGVmZi05NmI2LTQ2NjRmMjVkYzdkYSIsImMiOjEwfQ%3D%3D&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>""", unsafe_allow_html=True)

def model(label,X,y):
    model = []
    accuracy = []
    precision = []
    recall = []
    f1 = []

    filename = "Model/"+label+"_nb.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("Naive Bayes")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred))
    recall.append(recall_score(y, y_pred))
    f1.append(f1_score(y, y_pred))

    return pd.DataFrame({"Model":model,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-score":f1})

if uploaded_file:
    X = df.drop('delayStatus', axis = 1)
    y = df['delayStatus']

    st.dataframe(model("norm10",X,y))


