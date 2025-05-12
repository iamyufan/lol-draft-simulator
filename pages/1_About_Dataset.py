import streamlit as st
import pandas as pd
from data_processor import DataProcessor
import os
import plotly.express as px


@st.cache_data
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

    # Dataset Introduction
    st.header("Dataset Introduction")
    st.write("""
    This project utilizes the League of Legends dataset from Kaggle, which contains over 50,000 ranked games from the European West (EUW) server. 
    The dataset provides comprehensive information about each game, including champion picks, bans, and match outcomes.
    """)
    
    st.subheader("Dataset Source")
    st.write("""
    - **Source**: [League of Legends Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/league-of-legends)
    - **Region**: European West (EUW) server
    - **Game Type**: Ranked matches
    - **Size**: Over 50,000 games
    """)
    
    st.subheader("Dataset Contents")
    st.write("""
    The dataset includes:
    - Detailed match information for each game
    - Champion picks and bans for both teams
    - Game duration and outcome
    - JSON files for mapping champion and summoner spell IDs to their names
    """)

    processor = load_data()

    # Load the games data
    games_df = processor.games_df

    # Overview section
    st.header("Dataset Overview")
    st.write(f"Total number of games: {len(games_df)}")
    
    # Winner distribution
    st.subheader("Winner Distribution")
    winner_counts = games_df["winner"].value_counts()
    fig = px.pie(
        values=winner_counts.values,
        names=["Team 1", "Team 2"],
        title="Distribution of Winners"
    )
    st.plotly_chart(fig)
    
    # Unique values per column
    st.subheader("Unique Values per Column")
    exclude_cols = ["gameId", "creationTime", "gameDuration", "seasonId", "firstBlood", "firstTower", "firstInhibitor", "firstBaron", "firstDragon"]
    unique_counts = games_df.drop(columns=exclude_cols, errors="ignore").nunique()
    fig = px.bar(
        x=unique_counts.index,
        y=unique_counts.values,
        title="Unique Values per Column",
        labels={"x": "Column", "y": "Number of Unique Values"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Game duration distribution
    st.subheader("Game Duration Distribution")
    fig = px.histogram(
        games_df,
        x="gameDuration",
        #nbins=30,
        title="Distribution of Game Duration",
        labels={"gameDuration": "Game Duration (seconds)"}
    )
    st.plotly_chart(fig)
    
    # Game duration after filtering
    st.subheader("Game Duration Distribution (Filtered > 17 minutes)")
    normal_games = games_df[games_df["gameDuration"] >= 1020]  # 17 minutes in seconds
    durations = normal_games["gameDuration"]
    Q1 = durations.quantile(0.25)
    Q3 = durations.quantile(0.75)
    IQR = Q3 - Q1
    upper_fence = Q3 + 1.5 * IQR
    filtered_games = normal_games[durations <= upper_fence]
    fig = px.histogram(
        filtered_games,
        x="gameDuration",
        #nbins=30,
        title="Distribution of Filter Game Duration",
        labels={"gameDuration": "Game Duration (seconds)"}
    )
    st.plotly_chart(fig)
    # Top 10 most picked champions
    st.subheader("Top 10 Most Picked Champions")
    picked_champs = []
    for i in range(1, 6):
        picked_champs.extend(games_df[f"t1_champ{i}id"].tolist())
        picked_champs.extend(games_df[f"t2_champ{i}id"].tolist())
    
    champ_counts = pd.Series(picked_champs).value_counts().head(10)
    champ_names = [processor.get_champion_name(champ_id) for champ_id in champ_counts.index]
    
    fig = px.bar(
        x=champ_names,
        y=champ_counts.values,
        title="Top 10 Most Picked Champions",
        labels={"x": "Champion", "y": "Number of Games Picked"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    # Top 10 most banned champions
    st.subheader("Top 10 Most Banned Champions")
    banned_champs = []
    for i in range(1, 6):
        banned_champs.extend(games_df[f"t1_ban{i}"].tolist())
        banned_champs.extend(games_df[f"t2_ban{i}"].tolist())
    
    ban_counts = pd.Series(banned_champs).value_counts().head(10)
    ban_names = [processor.get_champion_name(champ_id) for champ_id in ban_counts.index]
    
    fig = px.bar(
        x=ban_names,
        y=ban_counts.values,
        title="Top 10 Most Banned Champions",
        labels={"x": "Champion", "y": "Number of Games Banned"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
if __name__ == "__main__":
    main()
