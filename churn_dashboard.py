import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

df = pd.read_csv("cleaned_churn_data.csv")
model = joblib.load("churn_prediction_model.pkl")

st.title("Customer Churn Prediction Dashboard")

total_customers = len(df)
churned = df[df["Churn"] == "Yes"].shape[0]
retained = df[df["Churn"] == "No"].shape[0]
churn_rate = churned / total_customers * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Retained Customers", retained)
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Churn Analysis",
    "Prediction",
    "Data Preview"
])

with tab1:
    churn_count = df["Churn"].value_counts().reset_index()
    churn_count.columns = ["Churn", "Count"]

    fig1 = px.pie(
        churn_count,
        values="Count",
        names="Churn",
        title="Churn Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        churn_count,
        x="Churn",
        y="Count",
        color="Churn",
        title="Churn Count"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(
        df,
        x="tenure",
        color="Churn",
        title="Tenure Distribution by Churn"
    )
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    st.subheader("Predict Customer Churn")

    gender = st.selectbox("Gender", df["gender"].unique())
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", df["Partner"].unique())
    dependents = st.selectbox("Dependents", df["Dependents"].unique())
    tenure = st.slider("Tenure", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", df["PhoneService"].unique())
    multiple_lines = st.selectbox("Multiple Lines", df["MultipleLines"].unique())
    internet_service = st.selectbox("Internet Service", df["InternetService"].unique())
    online_security = st.selectbox("Online Security", df["OnlineSecurity"].unique())
    online_backup = st.selectbox("Online Backup", df["OnlineBackup"].unique())
    device_protection = st.selectbox("Device Protection", df["DeviceProtection"].unique())
    tech_support = st.selectbox("Tech Support", df["TechSupport"].unique())
    streaming_tv = st.selectbox("Streaming TV", df["StreamingTV"].unique())
    streaming_movies = st.selectbox("Streaming Movies", df["StreamingMovies"].unique())
    contract = st.selectbox("Contract", df["Contract"].unique())
    paperless_billing = st.selectbox("Paperless Billing", df["PaperlessBilling"].unique())
    payment_method = st.selectbox("Payment Method", df["PaymentMethod"].unique())
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    if st.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Customer is likely to churn. Probability: {probability * 100:.2f}%")
        else:
            st.success(f"Customer is not likely to churn. Probability: {probability * 100:.2f}%")

with tab4:
    st.dataframe(df, use_container_width=True)