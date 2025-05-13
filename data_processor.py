import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple
#from collections import Counter
#from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataProcessor:
    def __init__(self, games_path: str = None, champion_info_path: str = None, champion_info_path_2: str = None):
        if games_path and champion_info_path:
            self.games_df = pd.read_csv(games_path)
            with open(champion_info_path, "r") as f:
                self.champion_info = json.load(f)["data"]
            
            with open(champion_info_path_2, "r") as f:
                self.champion_info_2 = json.load(f)["data"]
            
            self.all_tags = sorted({
                tag
                for info in self.champion_info_2.values()
                for tag in info.get("tags", [])
            })
            # Remove gameDuration < 17min
            self.games_df = self.games_df[self.games_df["gameDuration"] >= 1020]
            # Remove outliers
            # Calculate the IQR for gameDuration
            # IQR = Q3 - Q1
            durations = self.games_df["gameDuration"]
            Q1 = durations.quantile(0.25)
            Q3 = durations.quantile(0.75)
            IQR = Q3 - Q1
            upper_fence = Q3 + 1.5 * IQR
            self.games_df = self.games_df[durations <= upper_fence]
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
            'champion_info_2': self.champion_info_2,
            'all_tags': self.all_tags,
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
        processor.champion_info_2 = processor_state['champion_info_2']
        processor.all_tags = processor_state['all_tags']
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
        """Process the data to create features and labels.
        
        Returns:
            X: pd.DataFrame - Features
            y: pd.Series - Labels
        """
        # Create features for both teams
        team1_features = self._create_team_features("t1")
        team2_features = self._create_team_features("t2")
        team1_tag_counts = self._create_tag_count_features("t1")
        team2_tag_counts = self._create_tag_count_features("t2")
        # Combine features
        X = pd.concat([team1_features, team2_features, team1_tag_counts, team2_tag_counts], axis=1)
        y = self.games_df["winner"].map(
            {1: 1, 2: 0}
        )  # Convert to binary (1 = team1 wins, 0 = team2 wins)
        
        # Remove All 0 Columns
        # variances = X.var(axis=0)
        # zero_var = variances[variances == 0].index.tolist()
        # if zero_var:
        #     print(f"Dropping zero‐variance columns: {zero_var}")
        #     X = X.drop(columns=zero_var)
        # self._dropped_cols = zero_var

        return X, y

    def _create_tag_count_features(self, team_prefix: str):
        """
        count the number of each tag in each team
        for_example team1_tank_num, team2_fighter_num。
        """
        cols = [f"{team_prefix}_{tag.lower()}_num" for tag in self.all_tags]
        df = pd.DataFrame(0, index=self.games_df.index, columns=cols)

        for tag in self.all_tags:
            col = f"{team_prefix}_{tag.lower()}_num"
            def count_tag(row):
                cnt = 0
                for i in range(1, 6):
                    champ_id = row[f"{team_prefix}_champ{i}id"]
                    champ_key = self.champion_id_to_key.get(champ_id)
                    info = self.champion_info_2.get(champ_key, {})
                    if tag in info.get("tags", []):
                        cnt += 1
                return cnt
            df[col] = self.games_df.apply(count_tag, axis=1)

        return df
    
    def _create_team_features(self, team_prefix):
        """Create features for a specific team.
        
        Args:
            team_prefix: str - Prefix for the team (either "t1" or "t2")
            
        Returns:
            pd.DataFrame: DataFrame with features in the format expected by the model
        """
        # Initialize the feature columns
        feature_cols = [
            f"{champ_key}_picked_{team_prefix}"
            for champ_key in self.champion_id_to_key.values()
        ]
        # start all zeros
        feature_df = pd.DataFrame(0, index=self.games_df.index, columns=feature_cols)

        # for each pick‐slot, ADD into the correct champion column
        for i in range(1, 6):
            slot_col = f"{team_prefix}_champ{i}id"
            # vectorized per‐champ:
            for champ_id, champ_key in self.champion_id_to_key.items():
                mask = (self.games_df[slot_col] == champ_id)
                feature_df.loc[mask, f"{champ_key}_picked_{team_prefix}"] += 1

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
        #team1_bans,
        #team2_bans,
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
        for t in ("t1", "t2"):
            for champ_key in self.champion_key_to_id:
                feature_cols.append(f"{champ_key}_picked_{t}")
            for tag in self.all_tags:
                feature_cols.append(f"{t}_{tag.lower()}_num")

        features = pd.DataFrame(0, index=[0], columns=feature_cols)

        # accumulate picks for team1
        for champ_name in team1_champs or []:
            features.at[0, f"{champ_name}_picked_t1"] += 1

        # accumulate picks for team2
        for champ_name in team2_champs or []:
            features.at[0, f"{champ_name}_picked_t2"] += 1

        # # Process team 1 bans
        # for champ_name in team1_bans:
        #     if champ_name and champ_name != "":  # Skip None and empty strings
        #         champ_id = self.champion_key_to_id.get(champ_name)
        #         if champ_id:
        #             features[f"{champ_name}_banned_t1"] = 1

        # # Process team 2 bans
        # for champ_name in team2_bans:
        #     if champ_name and champ_name != "":  # Skip None and empty strings
        #         champ_id = self.champion_key_to_id.get(champ_name)
        #         if champ_id:
        #             features[f"{champ_name}_banned_t2"] = 1
        
        # team1 tag
        for champ_name in team1_champs or []:
            info = self.champion_info_2.get(champ_name, {})
            for tag in info.get("tags", []):
                col = f"t1_{tag.lower()}_num"
                features.at[0, col] += 1
        # team2 tag
        for champ_name in team2_champs or []:
            info = self.champion_info_2.get(champ_name, {})
            for tag in info.get("tags", []):
                col = f"t2_{tag.lower()}_num"
                features.at[0, col] += 1
        
        # Drop features that all 0 in train
        if hasattr(self, "_dropped_cols"):
            features = features.drop(columns=self._dropped_cols, errors="ignore")
        return self.scaler.transform(features)
