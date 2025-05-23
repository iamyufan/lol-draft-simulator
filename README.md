# League of Legends Draft Simulator

[[Live Demo](https://lol-draft-simulator.streamlit.app/)]

> Authors: Yufan Zhang, Ruitian Wu, Rundong Hu, Edward Wang

This project uses machine learning to predict the outcome of League of Legends matches based on champion picks, bans, and summoner spells. The dataset used is from the Kaggle dataset [(LoL) League of Legends Ranked Games](https://www.kaggle.com/datasets/datasnaek/league-of-legends).

## Project Structure

- `data_processor.py`: Handles data loading and preprocessing
- `train_model.py`: Trains and saves the prediction model
- `evaluate_model.py`: Evaluates model performance
- `recommend.py`: Make recommendations on the champion based on selected ones
- `app.py`: The home page for the Streamlit application
- `data/`: Contains the dataset and mapping files
- `checkpoint/`: Contains the data processor and model file
- `models/`: Stores trained models and encoders
- `pages/`: The streamlit application pages

## How to run the project

1. Clone the repository

```bash
git clone https://github.com/yourusername/league-of-legends-draft-simulator.git
```


2. Install the dependencies in a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the project

```bash
streamlit run app.py
```

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
   - Uses Logistic Regression, SVM, Random Forest and XGBoost
   - Key functionalities:
     - `train()`: Trains the model with certain model types and hyperparemeters
     - `predict()`: Makes binary predictions (team1/team2 win)
     - `predict_proba()`: Returns win probabilities
     - `save()/load()`: Model persistence

3. **Model Training (`train_model.py`)**

   - Orchestrates the training pipeline
   - Handles command-line arguments for customization
   - Saves both model and data processor for later use
   - Usage:

   ```bash
   python train_model.py --model_type xgboost --games_path data/games.csv --champion_info_path data/champion_info.json
   ```
   
4. **Recommend Champions (`recommend.py`)**

   - A script to show Top K recommended champions for certain teams with corresponding win rate
   - Input and adapt team composition and the integer k in the main function

5. **Web Application (`pages/1_Draft_Simulator.py`)**
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

