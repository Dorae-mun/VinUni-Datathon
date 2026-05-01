from pathlib import Path

import streamlit as st

from app.components.chatbot import render_chatbot
from src.data_loader import get_cached_data

css_path = Path(__file__).resolve().parents[1] / "style" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

data = get_cached_data()
render_chatbot(data)

st.components.v1.html(
    """
<iframe title="KTEAM - VinUni Datathon"
width="100%"
height="950"
src="https://app.powerbi.com/view?r=eyJrIjoiNTE0OGE5NDItZWM2MC00OWVhLTllZjUtYTJjMmE1ZDY2MDNkIiwidCI6IjQxYWI0MmE5LTM4MWItNDhjZi04YTg1LTcyMDQ2NDkyMjk3NiIsImMiOjEwfQ%3D%3D&pageName=ead47fd675acec0ebc82"
frameborder="0"
allowFullScreen="true">
</iframe>
""",
    height=950,
)