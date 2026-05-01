import streamlit as st

def kpi_card(title, value):
    st.metric(label=title, value=f"{value:,.0f}")