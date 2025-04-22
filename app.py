import streamlit as st

st.set_page_config(
    page_title="League of Legends Draft Helper",
    page_icon="🎮",
    layout="wide",
)

st.title("League of Legends Draft Helper")
st.write(
    """
Welcome to the League of Legends Draft Helper! This application helps you analyze and predict the outcome of League of Legends matches based on champion selections.

Use the sidebar to navigate between different pages:
- **Draft Simulator**: Simulate a draft and get win probability predictions
- **EDA**: Explore the dataset and view champion statistics
"""
)
