from typing import List, Tuple

def recommend_pick(
    team1_champs: List[str],
    team2_champs_partial: List[str],
    processor,
    model,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Recommend the final pick for team2 and return the win rate for team2.

    Parameters:
        team1_champs (List[str]): Full list of 5 champions for team1
        team2_champs_partial (List[str]): First 4 champions for team2
        processor: Trained DataProcessor
        model: Trained DraftPredictor model
        top_k (int): Number of recommendations to return

    Returns:
        List[Tuple[str, float]]: List of (champion_name, team2_win_probability), sorted descending
    """
    all_champs = list(processor.champion_key_to_id.keys())

    # Filter out already-picked champions
    used = set(team1_champs + team2_champs_partial)
    remaining = [c for c in all_champs if c not in used]

    recommendations = []

    # Try each remaining champion as the 5th pick for team2
    for champ in remaining:
        team2_full = team2_champs_partial + [champ]

        feats = processor.prepare_prediction_data(
            team1_champs=team1_champs,
            team2_champs=team2_full
        )

        prob_team1_win = model.predict_proba(feats)
        prob_team2_win = 1.0 - prob_team1_win

        recommendations.append((champ, prob_team2_win))

    # Sort by team2 win probability (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_k]

if __name__ == "__main__":
    from data_processor import DataProcessor
    from model import DraftPredictor

    PROC_PATH  = "checkpoints/draft_predictor_processor.joblib"
    CHAMP_JSON = "data/champion_info.json"
    MODEL_PATH = "checkpoints/draft_predictor.joblib"

    processor = DataProcessor.load(PROC_PATH)
    model = DraftPredictor(CHAMP_JSON)
    model.load(MODEL_PATH)

    team1 = ["Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh"]
    team2_partial = ["Ornn", "Lillia", "Orianna", "Aphelios"]

    top_recs = recommend_pick(team1, team2_partial, processor, model, top_k=8)

    print("Top 8 recommended 5th picks for team2 (sorted by win probability):\n")
    for i, (champ, prob) in enumerate(top_recs, 1):
        print(f"{i:>2}. {champ:<12}  →  team2 win rate: {prob:6.2%}")
