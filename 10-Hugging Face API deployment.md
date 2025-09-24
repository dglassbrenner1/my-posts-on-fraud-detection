---
layout: custom     # use your main layout
title: 10. Hugging Face API deployment         # page title
nav_order: 11
has_toc: true
nav_enabled: true
use_math: true
---

# 10. Hugging Face API deployment

We also deployed the tuned XGBoost in Hugging Face and created a Streamlit API to generate predictions for individual transactions.  


## 10.1 Generating the app 

We used the following app.py along with suitable Dockerfile, etc.


<details>
<summary>Click to expand/hide Hugging Face app.py code</summary>

<pre> ```python
import streamlit as st
import mlflow.sklearn
import pandas as pd

# Load model once (update model_uri accordingly)
model_uri = "./my_model" #"models:/workspace.default.fraud_detection_pipeline_model/1"
model = mlflow.sklearn.load_model(model_uri)

st.title("Fraud Detection Model Interface")
st.write("Enter transaction details below to get a fraud prediction:")

# Input fields matching your TransactionFeatures schema
TX_AMOUNT = st.number_input("Transaction Amount", min_value=0.0)
TX_DURING_WEEKEND = st.selectbox("Transaction During Weekend?", [0, 1])
TX_DURING_NIGHT = st.selectbox("Transaction During Night?", [0, 1])
Cust_Nb_Tx_1Day = st.number_input("Customer Number of Transactions in Last 1 Day", min_value=0)
Cust_Avg_Amt_1Day = st.number_input("Customer Avg Amount Last 1 Day", min_value=0.0)
Cust_Nb_Tx_7Day = st.number_input("Customer Number of Transactions in Last 7 Days", min_value=0)
Cust_Avg_Amt_7Day = st.number_input("Customer Avg Amount Last 7 Days", min_value=0.0)
Cust_Nb_Tx_30Day = st.number_input("Customer Number of Transactions in Last 30 Days", min_value=0)
Cust_Avg_Amt_30Day = st.number_input("Customer Avg Amount Last 30 Days", min_value=0.0)
Term_Nb_Tx_1Day = st.number_input("Terminal Number of Transactions in Last 1 Day", min_value=0)
Term_Risk_1Day = st.number_input("Terminal Risk Level Last 1 Day", min_value=0)
Term_Nb_Tx_7Day = st.number_input("Terminal Number of Transactions in Last 7 Days", min_value=0)
Term_Risk_7Day = st.number_input("Terminal Risk Level Last 7 Days", min_value=0)
Term_Nb_Tx_30Day = st.number_input("Terminal Number of Transactions in Last 30 Days", min_value=0)
Term_Risk_30Day = st.number_input("Terminal Risk Level Last 30 Days", min_value=0)

if st.button("Predict Fraud"):
    input_dict = {
        "TX_AMOUNT": TX_AMOUNT,
        "TX_DURING_WEEKEND": TX_DURING_WEEKEND,
        "TX_DURING_NIGHT": TX_DURING_NIGHT,
        "Cust_Nb_Tx_1Day": Cust_Nb_Tx_1Day,
        "Cust_Avg_Amt_1Day": Cust_Avg_Amt_1Day,
        "Cust_Nb_Tx_7Day": Cust_Nb_Tx_7Day,
        "Cust_Avg_Amt_7Day": Cust_Avg_Amt_7Day,
        "Cust_Nb_Tx_30Day": Cust_Nb_Tx_30Day,
        "Cust_Avg_Amt_30Day": Cust_Avg_Amt_30Day,
        "Term_Nb_Tx_1Day": Term_Nb_Tx_1Day,
        "Term_Risk_1Day": Term_Risk_1Day,
        "Term_Nb_Tx_7Day": Term_Nb_Tx_7Day,
        "Term_Risk_7Day": Term_Risk_7Day,
        "Term_Nb_Tx_30Day": Term_Nb_Tx_30Day,
        "Term_Risk_30Day": Term_Risk_30Day,
    }
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]  # Probability of class 1 (fraud)

    # st.success(f"Fraud Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
    st.write(f"Probability of fraud: {proba:.2%}")

``` </pre>
</details>

<br>

The Hugging Space app is live at: [text](https://huggingface.co/spaces/dglassbrenner/fraud_detection_api)

It is a simple Streamlit app where a user can enter values for the features of a transaction and select "Predict Fraud" to see the probability that a transaction with the selected features is fraudulent.
<br>


<img src="./images/Hugging Face API_page_1.png" alt="Hugging Face API_page_1" />

<br>


<img src="./images/Hugging Face API_page_2.png" alt="Hugging Face API_page_2" />



<br>

<table width="100%">
  <tr>
    <td align="left">
      <a href="/9-Databricks deployment.html">← Previous: 9. Databricks deployment</a>
    </td>
    <td align="right">
      Next: Post to come! →</a>
    </td>
  </tr>
</table>

