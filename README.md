# League of Legends Draft Simulator

This project uses machine learning to predict the outcome of League of Legends matches based on champion picks, bans, and summoner spells. The dataset used is from the Kaggle dataset [(LoL) League of Legends Ranked Games](https://www.kaggle.com/datasets/datasnaek/league-of-legends).

## Project Structure

- `data_processor.py`: Handles data loading and preprocessing
- `train_model.py`: Trains and saves the prediction model
- `evaluate_model.py`: Evaluates model performance
- `data/`: Contains the dataset and mapping files
- `models/`: Stores trained models and encoders

## Setup

1. Create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Make sure your data files are in the `data/` directory:
- `games.csv`: Match data
- `champion_info.json`: Champion ID to name mapping

## Usage

1. Train the model:
```bash
python train_model.py
```

After training, the model will be saved in the `checkpoints/` directory as `draft_predictor.joblib`.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Details

The model uses the following features:
- Team 1's 5 champions
- Team 2's 5 champions
- Team 1's 5 bans
- Team 2's 5 bans

The target variable is the winning team (1 for team 1, 0 for team 2).
