import argparse
import os
from data_processor import DataProcessor
from model import DraftPredictor
from sklearn.metrics import accuracy_score

def main():
    # Parse command line arguments
    # determine the directory this file lives in
    HERE = os.path.dirname(os.path.abspath(__file__))

    # build absolute defaults
    DEFAULT_GAMES   = os.path.join(HERE, 'data', 'games.csv')
    DEFAULT_CHAMPS  = os.path.join(HERE, 'data', 'champion_info.json')
    DEFAULT_CHAMPS2  = os.path.join(HERE, 'data', 'champion_info_2.json')
    parser = argparse.ArgumentParser(
        description='Train a League of Legends draft prediction model'
    )
    parser.add_argument(
        '--games_path',
        type=str,
        default=DEFAULT_GAMES,
        help='Path to the games data CSV file'
    )
    parser.add_argument(
        '--champion_info_path',
        type=str,
        default=DEFAULT_CHAMPS,
        help='Path to the champion info JSON file'
    )
    parser.add_argument(
        '--champion_info_path_2',
        type=str,
        default=DEFAULT_CHAMPS2,
        help='Path to the champion info 2 JSON file'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='draft_predictor',
        help='Name to save the model under'
    )
    args = parser.parse_args()

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    # Initialize data processor and process data
    print("Processing data...")
    data_processor = DataProcessor(args.games_path, args.champion_info_path, args.champion_info_path_2)
    X_train, X_test, y_train, y_test = data_processor.prepare_train_test_split(
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Initialize and train model
    print("Training model...")
    model = DraftPredictor(args.champion_info_path)
    model.train(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and processor
    model_path = os.path.join('checkpoints', f'{args.model_name}.joblib')
    processor_path = os.path.join('checkpoints', f'{args.model_name}_processor.joblib')
    
    model.save(model_path)
    data_processor.save(processor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Data processor saved to {processor_path}")

if __name__ == '__main__':
    main() 