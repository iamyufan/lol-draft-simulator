from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib
import pandas as pd
import numpy as np
from typing import List
import json


class DraftPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
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

        # Fit PCA on training data
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        print(f"Original number of features: {X_train.shape[1]}")
        print(f"Number of PCA components: {self.pca.n_components_}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2f}")

        self.model.fit(X_train_pca, y_train)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the winning team (1 for team1, 0 for team2)."""
        features_pca = self.pca.transform(features)
        return self.model.predict(features_pca)

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict the probability of team1 winning."""
        # Transform features using PCA
        features_pca = self.pca.transform(features)
        # Get probability of team1 winning
        prob = self.model.predict_proba(features_pca)[0][1]
        return prob

    def save(self, path: str):
        """Save the trained model to disk."""
        model_state = {
            'model': self.model,
            'pca': self.pca
        }
        joblib.dump(model_state, path)

    def load(self, path: str):
        """Load a trained model from disk."""
        model_state = joblib.load(path)
        self.model = model_state['model']
        self.pca = model_state['pca']
