"""Streamlit interface for delivery-time predictions."""

import streamlit as st


st.set_page_config(page_title="Food Delivery Time Prediction")
st.title("Food Delivery Time Prediction")
st.sidebar.header("Prediction inputs")
st.write("Enter delivery features to generate a prediction.")
