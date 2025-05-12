import streamlit as st
import sys
import os
import joblib
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor
from model import DraftPredictor
from recommend import recommend_pick

# Constants
PROC_PATH = "checkpoints/draft_predictor_processor.joblib"
CHAMP_JSON = "data/champion_info.json"
MODEL_PATH = "checkpoints/draft_predictor.joblib"

# Initialize session state for pre-filled champions
if "team1_champ_names" not in st.session_state:
    st.session_state.team1_champ_names = ["Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh"]
if "team2_champ_names" not in st.session_state:
    st.session_state.team2_champ_names = ["Ornn", "MasterYi", "Yasuo", "TahmKench"]

def load_models():
    """Load the trained models and processor."""
    processor = DataProcessor.load(PROC_PATH)
    model = DraftPredictor(CHAMP_JSON)
    model.load(MODEL_PATH)
    return processor, model

def get_all_champions(processor):
    """Get list of all champion names."""
    return sorted(processor.champion_key_to_id.keys())

def main():
    st.title("League of Legends Draft Recommendation")
    st.write("""
    This tool helps you make the optimal champion pick for your team based on the current draft state.
    Enter the champions for both teams and get recommendations for the final pick.
    """)

    # Load models
    try:
        processor, model = load_models()
        all_champions = get_all_champions(processor)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

    # Create two columns for team inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team 1 (Full Team)")
        team1_champs = []
        for i in range(5):
            default_index = (
                0
                if not st.session_state.team1_champ_names[i]
                else all_champions.index(st.session_state.team1_champ_names[i])
                + 1
            )
            champ = st.selectbox(
                f"Champion {i+1}",
                options=[""] + all_champions,
                key=f"team1_{i}",
                index=default_index
            )
            team1_champs.append(champ)
            st.session_state.team1_champ_names[i] = champ

    with col2:
        st.subheader("Team 2 (First 4 Picks)")
        team2_champs = []
        for i in range(4):
            default_index = (
                0
                if not st.session_state.team2_champ_names[i]
                else all_champions.index(st.session_state.team2_champ_names[i])
                + 1
            )
            champ = st.selectbox(
                f"Champion {i+1}",
                options=[""] + all_champions,
                key=f"team2_{i}",
                index=default_index
            )
            team2_champs.append(champ)
            st.session_state.team2_champ_names[i] = champ

    # Filter out empty selections
    team1_champs = [c for c in team1_champs if c]
    team2_champs = [c for c in team2_champs if c]

    # Add a button to get recommendations
    if st.button("Get Recommendations"):
        if len(team1_champs) != 5:
            st.error("Please select all 5 champions for Team 1")
        elif len(team2_champs) != 4:
            st.error("Please select all 4 champions for Team 2")
        else:
            try:
                # Get recommendations
                recommendations = recommend_pick(
                    team1_champs=team1_champs,
                    team2_champs_partial=team2_champs,
                    processor=processor,
                    model=model,
                    top_k=5
                )

                # Display recommendations
                st.subheader("Recommended Picks for Team 2")
                for i, (champ, prob) in enumerate(recommendations, 1):
                    st.write(f"{i}. **{champ}** - Win Probability: {prob:.1%}")

                # Display team compositions
                st.subheader("Current Team Compositions")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Team 1:**")
                    for champ in team1_champs:
                        st.write(f"- {champ}")
                with col2:
                    st.write("**Team 2:**")
                    for champ in team2_champs:
                        st.write(f"- {champ}")

            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")

if __name__ == "__main__":
    main() 