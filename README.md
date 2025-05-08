# League of Legends Draft Simulator

[[Live Demo](https://lol-draft-simulator.streamlit.app/)]

This project uses machine learning to predict the outcome of League of Legends matches based on champion picks, bans, and summoner spells. The dataset used is from the Kaggle dataset [(LoL) League of Legends Ranked Games](https://www.kaggle.com/datasets/datasnaek/league-of-legends).

## Project Structure

- `data_processor.py`: Handles data loading and preprocessing
- `train_model.py`: Trains and saves the prediction model
- `evaluate_model.py`: Evaluates model performance
- `data/`: Contains the dataset and mapping files
- `models/`: Stores trained models and encoders

## How to Contribute

### Project Architecture

The project is organized into several key components:

1. **Data Processing (`data_processor.py`)**

   - Handles data loading and preprocessing
   - Implements feature engineering for champion picks and bans
   - Provides data transformation for both training and inference
   - Key methods:
     - `prepare_train_test_split()`: Splits data into training and test sets
     - `process_data()`: Creates features from raw game data
     - `prepare_prediction_data()`: Transforms user input for model inference

2. **Model Implementation (`model.py`)**

   - Implements the `DraftPredictor` class
   - Uses Random Forest Classifier with PCA for dimensionality reduction
   - Key functionalities:
     - `train()`: Trains the model with PCA transformation
     - `predict()`: Makes binary predictions (team1/team2 win)
     - `predict_proba()`: Returns win probabilities
     - `save()/load()`: Model persistence

3. **Model Training (`train_model.py`)**

   - Orchestrates the training pipeline
   - Handles command-line arguments for customization
   - Saves both model and data processor for later use
   - Usage:

   ```bash
   python train_model.py --games_path data/games.csv --champion_info_path data/champion_info.json
   ```

4. **Web Application (`pages/1_Draft_Simulator.py`)**
   - Streamlit-based user interface
   - Implements champion selection and ban interface
   - Handles model loading and inference
   - Real-time win probability prediction

### Adding New Features

1. **New Data Processing Features**

   - Add new feature engineering methods to `DataProcessor` class
   - Update `process_data()` to include new features for training
   - Update `prepare_prediction_data()` to include new features for prediction
   - Ensure feature scaling is consistent

2. **Model Improvements**

   - Modify `DraftPredictor` class to include new model architectures
   - Update training pipeline in `train_model.py`
   - Add new evaluation metrics

3. **UI Enhancements**
   - Add new Streamlit components in `1_Draft_Simulator.py`
   - Implement additional visualization options
   - Add new user interaction features

### Best Practices

1. **Code Organization**

   - Keep data processing logic in `data_processor.py`
   - Maintain model-related code in `model.py`
   - Separate UI logic in Streamlit pages

2. **Model Management**
   - Save both model and processor together
   - Document model parameters and performance

## Model Details 

### Mid-Checkpoint Version

The model uses the following features:

- Team 1's 5 champions
- Team 2's 5 champions
- Team 1's 5 bans
- Team 2's 5 bans

The target variable is the winning team (1 for team 1, 0 for team 2).

#### Implementation Details

The model is implemented using a Random Forest Classifier with the following specifications:

- 100 decision trees (n_estimators=100)
- Random state set to 42 for reproducibility
- Uses scikit-learn's RandomForestClassifier implementation

The model provides the following key functionalities:

- `train(X, y)`: Trains the model on the given features and target data
- `predict(features)`: Predicts the winning team (1 for team1, 0 for team2)
- `predict_proba(features)`: Returns the probability of team1 winning
- `save(path)`: Saves the trained model to disk
- `load(path)`: Loads a trained model from disk

The model also includes champion information handling:

- Loads champion data from `data/champion_info.json`
- Maintains a mapping between champion IDs and their names
- Filters out invalid champion entries (where key is "None")
