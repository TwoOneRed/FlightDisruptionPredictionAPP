import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pickle

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
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
    st.text("Uploaded Dataset")
    st.dataframe(df.head(5))
    sheet = client.open("FYP2Data_PowerBI").sheet1
    # Clear the existing data from the sheet
    sheet.clear()

    #Convert To String
    newdata = df.astype(str)

    # Update the Google Sheets file with the modified DataFrame
    st.write(sheet.update([newdata.columns.values.tolist()] + newdata.values.tolist()))

else:
    st.text("Original Dataset")
    sheet = client.open("FYP2Data_PowerBI").sheet1
    df = sheet.get_all_values()
    # Check if there are any rows in the DataFrame
    if len(df) > 1:
        df = pd.DataFrame(df[1:], columns=df[0])
        st.dataframe(df)
    else:
        st.text("No data available in the spreadsheet")

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
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))

    filename = "Model/"+label+"_svm.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("Support Vector Machine")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))

    filename = "Model/"+label+"_dt.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("Decision Tree")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))

    filename = "Model/"+label+"_rf.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("Random Forest")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))

    filename = "Model/"+label+"_knn.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("K Nearest Neighbours")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))

    filename = "Model/"+label+"_lr.pkl"
    loaded_model = pickle.load(open(filename,'rb'))
    y_pred = loaded_model.predict(X)
    model.append("Logistic Regression")
    accuracy.append(round((accuracy_score(y, y_pred)*100), 2))
    precision.append(precision_score(y, y_pred, zero_division=0, average='weighted'))
    recall.append(recall_score(y, y_pred, average='weighted'))
    f1.append(f1_score(y, y_pred, average='weighted'))


    return pd.DataFrame({"Model":model,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-score":f1})


rfe_score=pd.read_csv("Model/rfe_score.csv")

if len(df) > 1:
    df = df.drop(columns=['dep_Lat','dep_Lon','arr_Lat','arr_Lon','delayed'])
    data_encode = df.copy()
    data_encode = data_encode.apply(LabelEncoder().fit_transform)

    X = data_encode.drop('delayStatus', axis = 1)
    X10 = X[rfe_score.Features[0:10]].copy()
    X30 = X[rfe_score.Features[0:30]].copy()
    y = data_encode['delayStatus']

    st.title("Prediction Result Using Pre-Trained Model")
    st.text("Model Trained Using top 10 Features and Normal Data")
    st.dataframe(model("norm10",X10,y))

    st.text("Model Trained Using top 30 Features and Normal Data")
    st.dataframe(model("norm30",X30,y))

    st.text("Model Trained Using top 10 Features and SMOTE Data")
    st.dataframe(model("smote10",X10,y))

    st.text("Model Trained Using top 30 Features and SMOTE Data")
    st.dataframe(model("smote30",X30,y))



