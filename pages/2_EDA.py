import streamlit as st
import pandas as pd
from data_processor import DataProcessor
import os
import matplotlib.pyplot as plt


@st.cache_resource
def load_data():
    processor_path = os.path.join("checkpoints", "draft_predictor_processor.joblib")
    if not os.path.exists(processor_path):
        st.error(
            f"Data processor not found at {processor_path}. Please train the model first using train_model.py"
        )
        st.stop()
    return DataProcessor.load(processor_path)


def main():
    st.title("League of Legends Draft Analysis")

    processor = load_data()

    # Load the games data
    games_df = processor.games_df

    # Overview section
    st.header("Dataset Overview")
    st.write(f"Total number of games: {len(games_df)}")
    
    # Winner distribution
    st.subheader("Winner Distribution")
    winner_counts = games_df["winner"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        winner_counts.values,
        labels=["Team 1", "Team 2"],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title("Distribution of Winners")
    st.pyplot(fig)

    # Game duration distribution
    st.subheader("Game Duration Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(games_df["gameDuration"], bins=50)
    ax.set_xlabel("Game Duration (seconds)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Distribution of Game Duration")
    st.pyplot(fig)
    
    # Top 10 most picked champions
    st.subheader("Top 10 Most Picked Champions")
    picked_champs = []
    for i in range(1, 6):
        picked_champs.extend(games_df[f"t1_champ{i}id"].tolist())
        picked_champs.extend(games_df[f"t2_champ{i}id"].tolist())
    
    champ_counts = pd.Series(picked_champs).value_counts().head(10)
    champ_names = [processor.get_champion_name(champ_id) for champ_id in champ_counts.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(champ_names, champ_counts.values)
    ax.set_xlabel("Champion")
    ax.set_ylabel("Number of Picks")
    ax.set_title("Top 10 Most Picked Champions")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Top 10 most banned champions
    st.subheader("Top 10 Most Banned Champions")
    banned_champs = []
    for i in range(1, 6):
        banned_champs.extend(games_df[f"t1_ban{i}"].tolist())
        banned_champs.extend(games_df[f"t2_ban{i}"].tolist())
    
    ban_counts = pd.Series(banned_champs).value_counts().head(10)
    ban_names = [processor.get_champion_name(champ_id) for champ_id in ban_counts.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(ban_names, ban_counts.values)
    ax.set_xlabel("Champion")
    ax.set_ylabel("Number of Bans")
    ax.set_title("Top 10 Most Banned Champions")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
