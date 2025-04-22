import streamlit as st
import pandas as pd
from data_processor import DataProcessor
import os
import plotly.express as px


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
    fig = px.pie(
        values=winner_counts.values,
        names=["Team 1", "Team 2"],
        title="Distribution of Winners"
    )
    st.plotly_chart(fig)

    # Game duration distribution
    st.subheader("Game Duration Distribution")
    fig = px.histogram(
        games_df,
        x="gameDuration",
        nbins=50,
        title="Distribution of Game Duration",
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
