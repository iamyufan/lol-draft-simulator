from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np
from typing import List
import json


class DraftPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Load champion info
        with open("data/champion_info.json", "r") as f:
            self.champion_info = json.load(f)["data"]
        # Create champion ID to name mapping
        self.id_to_name = {
            int(champ["id"]): champ["key"]
            for champ in self.champion_info.values()
            if champ["key"] != "None"
        }

    def train(self, X: np.ndarray, y: pd.Series):
        """Train the model on the given data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the winning team (1 for team1, 0 for team2)."""
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict the probability of team1 winning."""
        # Get probability of team1 winning
        prob = self.model.predict_proba(features)[0][1]
        return prob

    def save(self, path: str):
        """Save the trained model to disk."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load a trained model from disk."""
        self.model = joblib.load(path)
