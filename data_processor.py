import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataProcessor:
    def __init__(self, games_path: str = None, champion_info_path: str = None):
        if games_path and champion_info_path:
            self.games_df = pd.read_csv(games_path)
            with open(champion_info_path, "r") as f:
                self.champion_info = json.load(f)["data"]
        else:
            self.games_df = None
            self.champion_info = None
        
        # Initialize scaler
        self.scaler = StandardScaler()

        if games_path and champion_info_path:
            # Create mappings
            # Create a list of all champion keys from the dictionary
            champion_keys = [
                champion["key"]
                for champion in self.champion_info.values()
                if champion["key"] != "None"
            ]
            champion_keys.sort()

            print(f"Number of champions: {len(champion_keys)}")
            print(f"Number of games: {len(self.games_df)}")

            # Create mappings
            self.champion_id_to_key = {
                champion["id"]: champion["key"]
                for champion in self.champion_info.values()
                if champion["key"] != "None"
            }
            self.champion_key_to_id = {
                champion["key"]: champion["id"]
                for champion in self.champion_info.values()
                if champion["key"] != "None"
            }

    def save(self, path: str):
        """Save the data processor with its fitted scaler."""
        # Create a dictionary of attributes to save
        processor_state = {
            'champion_info': self.champion_info,
            'champion_id_to_key': self.champion_id_to_key,
            'champion_key_to_id': self.champion_key_to_id,
            'scaler': self.scaler,
            'games_df': self.games_df
        }
        joblib.dump(processor_state, path)

    @classmethod
    def load(cls, path: str):
        """Load a saved data processor."""
        processor_state = joblib.load(path)
        
        # Create a new instance
        processor = cls()
        
        # Restore the state
        processor.champion_info = processor_state['champion_info']
        processor.champion_id_to_key = processor_state['champion_id_to_key']
        processor.champion_key_to_id = processor_state['champion_key_to_id']
        processor.scaler = processor_state['scaler']
        processor.games_df = processor_state['games_df']
        
        return processor

    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        X, y = self.process_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Fit and transform the training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def process_data(self):
        # Create features for both teams
        team1_features = self._create_team_features("t1")
        team2_features = self._create_team_features("t2")

        # Combine features
        X = pd.concat([team1_features, team2_features], axis=1)
        y = self.games_df["winner"].map(
            {1: 1, 2: 0}
        )  # Convert to binary (1 = team1 wins, 0 = team2 wins)

        return X, y

    def _create_team_features(self, team_prefix):
        # Initialize the feature columns
        feature_cols = []
        for champ_id, champ_key in self.champion_id_to_key.items():
            feature_cols.append(f"{champ_key}_picked_{team_prefix}")
            feature_cols.append(f"{champ_key}_banned_{team_prefix}")

        # Initialize the X DataFrame
        feature_df = pd.DataFrame(0, index=self.games_df.index, columns=feature_cols)

        # Process champions
        for i in range(1, 6):
            for champ_id, champ_key in self.champion_id_to_key.items():
                feature_df[f"{champ_key}_picked_{team_prefix}"] = self.games_df[
                    f"{team_prefix}_champ{i}id"
                ].apply(lambda x: 1 if x == champ_id else 0)
                feature_df[f"{champ_key}_banned_{team_prefix}"] = self.games_df[
                    f"{team_prefix}_ban{i}"
                ].apply(lambda x: 1 if x == champ_id else 0)

        return feature_df

    def get_champion_name(self, champion_id: int) -> str:
        """Convert champion ID to name."""
        return self.champion_id_to_key.get(champion_id, "Unknown")

    def get_all_champions(self) -> List[Dict[str, str]]:
        """Get list of all champions with their IDs and names."""
        return [
            {"id": str(id), "name": name}
            for id, name in self.champion_id_to_key.items()
        ]

    def prepare_prediction_data(
        self,
        team1_champs,
        team2_champs,
        team1_bans,
        team2_bans,
    ):
        """Prepare data for prediction from champion selections and bans.

        Args:
            team1_champs: List of champion names for team 1
            team2_champs: List of champion names for team 2
            team1_bans: List of champion names banned by team 1
            team2_bans: List of champion names banned by team 2

        Returns:
            pd.DataFrame: DataFrame with features in the format expected by the model
        """
        # Create feature columns
        feature_cols = []
        for team_prefix in ["t1", "t2"]:
            for champ_key in self.champion_key_to_id.keys():
                feature_cols.append(f"{champ_key}_picked_{team_prefix}")
                feature_cols.append(f"{champ_key}_banned_{team_prefix}")

        # Initialize DataFrame with zeros
        features = pd.DataFrame(0, index=[0], columns=feature_cols)

        # Process team 1 champions
        for champ_name in team1_champs:
            if champ_name and champ_name != "":  # Skip None and empty strings
                champ_id = self.champion_key_to_id.get(champ_name)
                if champ_id:
                    features[f"{champ_name}_picked_t1"] = 1

        # Process team 2 champions
        for champ_name in team2_champs:
            if champ_name and champ_name != "":  # Skip None and empty strings
                champ_id = self.champion_key_to_id.get(champ_name)
                if champ_id:
                    features[f"{champ_name}_picked_t2"] = 1

        # Process team 1 bans
        for champ_name in team1_bans:
            if champ_name and champ_name != "":  # Skip None and empty strings
                champ_id = self.champion_key_to_id.get(champ_name)
                if champ_id:
                    features[f"{champ_name}_banned_t1"] = 1

        # Process team 2 bans
        for champ_name in team2_bans:
            if champ_name and champ_name != "":  # Skip None and empty strings
                champ_id = self.champion_key_to_id.get(champ_name)
                if champ_id:
                    features[f"{champ_name}_banned_t2"] = 1

        # Apply the same scaling as training data
        features_scaled = self.scaler.transform(features)

        return features_scaled
