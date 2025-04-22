import streamlit as st
import pandas as pd
from data_processor import DataProcessor
from model import DraftPredictor
import os

# Initialize session state
if "team1_champ_names" not in st.session_state:
    st.session_state.team1_champ_names = ["LeeSin", "Ahri", "Jinx", "Thresh", "Gnar"]
if "team2_champ_names" not in st.session_state:
    st.session_state.team2_champ_names = ["Zed", "Yasuo", "Darius", "Kayn", "Vayne"]
if "team1_ban_names" not in st.session_state:
    st.session_state.team1_ban_names = [
        "Lucian",
        "Nami",
        "Aatrox",
        "Orianna",
        "Sejuani",
    ]
if "team2_ban_names" not in st.session_state:
    st.session_state.team2_ban_names = [
        "Ezreal",
        "Lulu",
        "Graves",
        "Syndra",
        "Renekton",
    ]

# Initialize data processor and model
@st.cache_resource
def load_data():
    processor_path = os.path.join("checkpoints", "draft_predictor_processor.joblib")
    if not os.path.exists(processor_path):
        st.error(
            f"Data processor not found at {processor_path}. Please train the model first using train_model.py"
        )
        st.stop()
    return DataProcessor.load(processor_path)

@st.cache_resource
def load_model():
    model = DraftPredictor()
    model_path = os.path.join("checkpoints", "draft_predictor.joblib")
    if not os.path.exists(model_path):
        st.error(
            f"Model not found at {model_path}. Please train the model first using train_model.py"
        )
        st.stop()
    model.load(model_path)
    return model

def main():
    st.title("League of Legends Draft Simulator")
    
    processor = load_data()
    predictor = load_model()

    # Get list of all champions
    champions = processor.get_all_champions()
    champion_options = {champ["name"]: int(champ["id"]) for champ in champions}

    # Create two columns for team selection
    col1, col2 = st.columns(2)

    with col1:
        st.header("Team 1")
        for i in range(5):
            default_index = (
                0
                if not st.session_state.team1_champ_names[i]
                else list(champion_options.keys()).index(
                    st.session_state.team1_champ_names[i]
                )
                + 1
            )
            st.session_state.team1_champ_names[i] = st.selectbox(
                f"Team 1 Champion {i+1}",
                options=[""] + list(champion_options.keys()),
                key=f"team1_champ_{i}",
                index=default_index,
            )
        st.subheader("Team 1 Bans")
        for i in range(5):
            default_index = (
                0
                if not st.session_state.team1_ban_names[i]
                else list(champion_options.keys()).index(
                    st.session_state.team1_ban_names[i]
                )
                + 1
            )
            st.session_state.team1_ban_names[i] = st.selectbox(
                f"Team 1 Ban {i+1}",
                options=[""] + list(champion_options.keys()),
                key=f"team1_ban_names_{i}",
                index=default_index,
            )

    with col2:
        st.header("Team 2")
        for i in range(5):
            default_index = (
                0
                if not st.session_state.team2_champ_names[i]
                else list(champion_options.keys()).index(
                    st.session_state.team2_champ_names[i]
                )
                + 1
            )
            st.session_state.team2_champ_names[i] = st.selectbox(
                f"Team 2 Champion {i+1}",
                options=[""] + list(champion_options.keys()),
                key=f"team2_champ_{i}",
                index=default_index,
            )
        st.subheader("Team 2 Bans")
        for i in range(5):
            default_index = (
                0
                if not st.session_state.team2_ban_names[i]
                else list(champion_options.keys()).index(st.session_state.team2_ban_names[i])
                + 1
            )
            st.session_state.team2_ban_names[i] = st.selectbox(
                f"Team 2 Ban {i+1}",
                options=[""] + list(champion_options.keys()),
                key=f"team_ban_name_{i}",
                index=default_index,
            )

    # Add predict button
    if st.button("Predict Match Outcome"):
        features = processor.prepare_prediction_data(
            st.session_state.team1_champ_names,
            st.session_state.team2_champ_names,
            st.session_state.team1_ban_names,
            st.session_state.team2_ban_names,
        )
        # Get prediction
        prob = predictor.predict_proba(features)

        # Display prediction
        st.subheader("Prediction")
        st.write(f"Team 1 Win Probability: {prob:.2%}")
        st.write(f"Team 2 Win Probability: {(1-prob):.2%}")

        # Visualize prediction
        st.progress(prob)

if __name__ == "__main__":
    main() 