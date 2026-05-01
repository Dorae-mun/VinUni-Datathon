import streamlit as st


def date_filter(df):
    date_col = "Date" if "Date" in df.columns else "date"
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    return st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
    )
