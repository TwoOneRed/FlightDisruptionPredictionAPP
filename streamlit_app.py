import streamlit as st
import pandas as pd

from powerbiclient import Report

st.title('Flight Disruption Prediction Applications')
st.text('1191100292  Leong Yi Hong')


# Define your Azure AD and Power BI details
tenant_id = '7419ae17-5bc9-4dde-a96b-50878d8575f4'
client_id = 'ad3b6b53-2382-418f-8014-13aeaf79205e'
client_secret = 'kxc8Q~UJpDQZEHdVOrVaJBH0He5rhvWgwFhHLbKK'
authority = f"https://login.microsoftonline.com/{tenant_id}"

group_id = "f4b32c96-7686-4187-ba83-59ad83d16377"
report_id = 'ff3647ff-cd01-4251-b793-64f480078853'  # Replace with your actual report ID

# Create an instance of the Report class
report = Report(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret, authority=authority, report_id=report_id, group_id=group_id)

# Display the report
st.markdown(report,unsafe_allow_html=True)