import streamlit as st
from src.predict import predict_condition

st.title("RespireSense – Respiratory Detection")

resp = st.number_input("Respiratory Rate")
spo2 = st.number_input("Oxygen Saturation")
ph = st.number_input("Blood pH")

if st.button("Predict"):

    result = predict_condition(resp,spo2,ph)

    st.success(f"Condition: {result}")
