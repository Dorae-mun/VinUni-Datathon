import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Datathon 2026 Dashboard",
    layout="wide",
    page_icon="Dashboard",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "style" / "custom.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 0.4rem;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.switch_page("pages/1_overview.py")
