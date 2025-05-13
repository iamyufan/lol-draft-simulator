import argparse
import os
from data_processor import DataProcessor
from model import DraftPredictor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
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
        '--model_type',
        type=str,
        default='xgboost',
        choices=['logistic_regression', 'svm', 'random_forest', 'xgboost'],
        help='Which algorithm to use'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='draft_predictor',
        help='Name to save the model under'
    )
    
    # RF & XGB
    parser.add_argument('--n_estimators',   type=int,   nargs='+', default=[50,100])
    parser.add_argument('--max_depth',      type=int,   nargs='+', default=[5,10])
    parser.add_argument('--learning_rate',  type=float, nargs='+', default=[0.1,0.3])
    # LR & SVM
    parser.add_argument('--C',              type=float, nargs='+', default=[0.01,0.1,1,10])
    parser.add_argument('--kernel',         type=str,   nargs='+', default=['linear','rbf'])
    parser.add_argument('--gamma',          type=str,   nargs='+', default=['scale','auto'])
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
    print("Data processing complete.")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    grid = []
    if args.model_type in ('random_forest','xgboost'):
        for n in args.n_estimators:
            for d in args.max_depth:
                for lr in (args.learning_rate if args.model_type=='xgboost' else [None]):
                    params = {'n_estimators':n, 'max_depth':d}
                    if lr is not None: params['learning_rate'] = lr
                    grid.append(params)

    elif args.model_type == 'logistic_regression':
        for C in args.C:
            grid.append({'C':C})

    elif args.model_type == 'svm':
        for C in args.C:
            for kernel in args.kernel:
                for gamma in args.gamma:
                    grid.append({'C':C, 'kernel':kernel, 'gamma':gamma})
                    
                    
    # 3) sweep
    best_acc, best_params = 0.0, None
    for params in grid:
        print("→ trying", args.model_type, params)
        model = DraftPredictor(
            model_type=args.model_type,
            **params,champion_info_path=args.champion_info_path
        )
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("   ↳ acc =", acc)
        if acc > best_acc:
            best_acc, best_params = acc, params.copy()

    print(f"Best validation accuracy = {best_acc:.4f} with {best_params}")

    # Initialize and train model
    print(f"Training a {args.model_type} model...")
    final_model = DraftPredictor(model_type=args.model_type, **best_params,champion_info_path=args.champion_info_path)
    final_model.train(X_train, y_train)

    # 5) evaluate on test set
    y_pred      = final_model.predict(X_test)
    print("Test set predictions:", y_pred)
    # for AUC we need scores / probabilities if available
    try:
        y_score = final_model.model.predict_proba(X_test)[:,1]
    except AttributeError:
        y_score = final_model.model.decision_function(X_test)
    print("\n=== TEST SET METRICS ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("AUC-ROC  :", roc_auc_score(y_test, y_score))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    

    # Save model and processor
    model_path = os.path.join('checkpoints', f'{args.model_name}.joblib')
    processor_path = os.path.join('checkpoints', f'{args.model_name}_processor.joblib')
    
    model.save(model_path)
    data_processor.save(processor_path)
    
    print(f"Model saved to {model_path}")
    print(f"Data processor saved to {processor_path}")

if __name__ == '__main__':
    main() 