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

else:
    st.text("Original Dataset")
    sheet = client.open("FYP2_PredictionResult").sheet1
    df = sheet.get_all_values()
    # Check if there are any rows in the DataFrame
    if len(df) > 1:
        df = pd.DataFrame(df[1:], columns=df[0])
        st.dataframe(df)
    else:
        st.text("No data available in the spreadsheet")

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

st.title("Prediction Result Using Pre-Trained Model")

rfe_score=pd.read_csv("Model/rfe_score.csv")

merged_df = pd.DataFrame()

if len(df) > 1:
    data_encode = df.drop(columns=['dep_Lat','dep_Lon','arr_Lat','arr_Lon','delayed','Prediction']).copy()
    data_encode = data_encode.apply(LabelEncoder().fit_transform)

    X = data_encode.drop('delayStatus', axis = 1)
    X10 = X[rfe_score.Features[0:10]].copy()
    X30 = X[rfe_score.Features[0:30]].copy()
    y = data_encode['delayStatus']
    
    dataframes = []

    st.text("Model Trained Using top 10 Features and Normal Data")
    norm10result = model("norm10",X10,y)
    st.dataframe(norm10result)
    norm10result["Model_Data"] = "norm10"
    dataframes.append(norm10result)

    st.text("Model Trained Using top 30 Features and Normal Data")
    norm30result = model("norm30",X30,y)
    st.dataframe(norm30result)
    norm30result["Model_Data"] = "norm30"
    dataframes.append(norm30result)

    st.text("Model Trained Using top 10 Features and SMOTE Data")
    smote10result = model("smote10",X10,y)
    st.dataframe(smote10result)
    smote10result["Model_Data"] = "smote10"
    dataframes.append(smote10result)

    st.text("Model Trained Using top 30 Features and SMOTE Data")
    smote30result = model("smote30",X30,y)
    st.dataframe(smote30result)
    smote30result["Model_Data"] = "smote30"
    dataframes.append(smote30result)

    # Merge all the dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    highest_accuracy_row = merged_df.loc[merged_df['Accuracy'].idxmax()]
    text = ""
    if (highest_accuracy_row['Model_Data'] == "norm10"):
        text = "Normal Data with 10 Features"
    elif (highest_accuracy_row['Model_Data'] == "norm30"):
        text = "Normal Data with 30 Features"    
    elif (highest_accuracy_row['Model_Data'] == "smote10"):
        text = "SMOTE Data with 10 Features"        
    else:
        text = "SMOTE Data with 30 Features"        

    st.text(highest_accuracy_row["Model"]+ " using "+text+" pre-trained model has the\nhighest accuracy of "
            +str(highest_accuracy_row['Accuracy'])+" compare to other model. \nTherefore, "
            +highest_accuracy_row["Model"]+" is used in the process afterwards.")

    predict_encode = df.drop(columns=['dep_Lat','dep_Lon','arr_Lat','arr_Lon','delayed','Prediction']).copy()
    predict_encode = predict_encode.apply(LabelEncoder().fit_transform)
    X = predict_encode.drop('delayStatus', axis = 1)
    X10 = X[rfe_score.Features[0:10]].copy()
    X30 = X[rfe_score.Features[0:30]].copy()
    
    if (highest_accuracy_row["Model"] == "Naive Bayes"):
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_nb.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred

    elif (highest_accuracy_row["Model"] == "Support Vector Machine"):
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_svm.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred

    elif (highest_accuracy_row["Model"] == "Decision Tree"):
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_dt.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred

    elif (highest_accuracy_row["Model"] == "Random Forest"):
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_rf.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred

    elif (highest_accuracy_row["Model"] == "K Nearest Neighbours"):
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_knn.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred

    else:
        filename = "Model/"+highest_accuracy_row['Model_Data']+"_lr.pkl"
        loaded_model = pickle.load(open(filename,'rb'))
        if(highest_accuracy_row['Model_Data'] == "norm10" or highest_accuracy_row['Model_Data'] == "smote10"):
            y_pred = loaded_model.predict(X10)    
        else:
            y_pred = loaded_model.predict(X30)    
        df['Prediction'] = y_pred
    
    sheet = client.open("FYP2_PredictionResult").sheet1
    #Clear the existing data from the sheet
    sheet.clear()

    #Convert To String
    df = df.astype(str)

    # Update the Google Sheets file with the modified DataFrame
    st.write(sheet.update([df.columns.values.tolist()] + df.values.tolist()))

else:
    st.text("No data available in the spreadsheet")

# Create a Streamlit app
st.title("Power BI Dashboard")

# Display the report
st.markdown("""<iframe title="PB" width="900" height="550" src="https://app.powerbi.com/view?r=eyJrIjoiMmU3ZjFmODItMDE1My00ZDE4LWJhNmQtOTFiYmM1ODAxYWU4IiwidCI6IjdlMGI1ZmNmLTEyYzQtNGVmZi05NmI2LTQ2NjRmMjVkYzdkYSIsImMiOjEwfQ%3D%3D&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>""", unsafe_allow_html=True)


