"""
Experiment Runner: Test models across multiple season splits
Aggregates results for robust evaluation
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss, classification_report
import pickle
import json
import os
from datetime import datetime
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments across multiple season splits')
    
    parser.add_argument('--model', type=str, default='conversion', choices=['conversion', 'fg', 'both'])
    parser.add_argument('--n-runs', type=int, default=10, help='Number of different train/test splits')
    parser.add_argument('--test-size', type=float, default=0.15, help='Proportion of seasons for testing')
    parser.add_argument('--data-path', type=str, default='data/filtered_data/fourth_downs_with_features.csv')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Where to save results')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for this experiment')
    
    # XGBoost params (can test different configs)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    parser.add_argument('--random-seed', type=int, default=42)
    
    return parser.parse_args()


def load_conversion_data(df):
    """Load and prepare conversion model data"""
    went_for_it = df[df['went_for_it'] == 1].copy()
    
    features = [
        'yards_to_go', 'field_position', 'score_diff', 'time_remaining',
        'quarter', 'is_home', 'in_red_zone', 'in_fg_range', 'in_own_territory',
        'short_yardage', 'very_short', 'long_distance', 'losing', 'winning',
        'tied', 'close_game', 'late_game', 'final_2_min', 'desperate_situation',
        'is_dome', 'is_outdoors', 'cold_weather', 'windy'
    ]
    
    available = [f for f in features if f in went_for_it.columns]
    clean_data = went_for_it[available + ['converted', 'season']].dropna()
    
    return clean_data[available], clean_data['converted'], clean_data


def load_fg_data(df):
    """Load and prepare FG model data"""
    fg_attempts = df[df['is_field_goal'] == 1].copy()
    
    if 'fg_distance' not in fg_attempts.columns:
        fg_attempts['fg_distance'] = fg_attempts['field_position'] + 17
    
    features = [
        'fg_distance', 'is_home', 'quarter', 'score_diff',
        'is_dome', 'is_outdoors', 'cold_weather', 'windy',
        'late_game', 'final_2_min'
    ]
    
    available = [f for f in features if f in fg_attempts.columns]
    clean_data = fg_attempts[available + ['fg_made', 'season']].dropna()
    
    return clean_data[available], clean_data['fg_made'], clean_data


def create_season_split(seasons, test_size, random_state=None):
    """
    Randomly select seasons for train/test split
    
    Args:
        seasons: List of all available seasons
        test_size: Proportion of seasons to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_seasons, test_seasons
    """
    if random_state is not None:
        random.seed(random_state)
    
    seasons_list = list(seasons)
    n_test = max(1, int(len(seasons_list) * test_size))
    
    test_seasons = random.sample(seasons_list, n_test)
    train_seasons = [s for s in seasons_list if s not in test_seasons]
    
    return train_seasons, test_seasons


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    
    Returns dict of metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'n_samples': len(y_test),
        'positive_rate': y_test.mean()
    }
    
    return metrics


def run_single_experiment(X, y, clean_data, train_seasons, test_seasons, model_params, model_name):
    """
    Run a single train/test experiment
    
    Returns metrics dict
    """
    # Split by season
    train_mask = clean_data['season'].isin(train_seasons)
    test_mask = clean_data['season'].isin(test_seasons)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Train model
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Add split info
    metrics['train_seasons'] = train_seasons
    metrics['test_seasons'] = test_seasons
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)
    
    return metrics, model


def aggregate_results(all_results):
    """
    Aggregate metrics across multiple runs
    
    Returns summary statistics
    """
    metrics_to_aggregate = ['accuracy', 'auc_roc', 'brier_score', 'log_loss']
    
    aggregated = {}
    
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in all_results]
        
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'values': values  # Keep individual values
        }
    
    return aggregated


def save_results(results, args, output_dir):
    """
    Save experiment results to JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment metadata
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': args.experiment_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'model_type': args.model,
        'n_runs': args.n_runs,
        'test_size': args.test_size,
        'hyperparameters': {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree
        },
        'aggregated_metrics': results['aggregated'],
        'individual_runs': results['individual_runs']
    }
    
    # Save to JSON
    filename = f"{experiment_data['experiment_name']}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")
    
    return filepath


def print_results(aggregated):
    """
    Print aggregated results in nice format
    """
    print("\n" + "="*70)
    print("AGGREGATED RESULTS")
    print("="*70)
    
    for metric, stats in aggregated.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Median: {stats['median']:.4f}")
        
        # 95% confidence interval (mean ± 2*std)
        ci_lower = stats['mean'] - 2*stats['std']
        ci_upper = stats['mean'] + 2*stats['std']
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


def main():
    args = parse_args()
    
    print("="*70)
    print(f"EXPERIMENT RUNNER: {args.model.upper()} MODEL")
    print("="*70)
    print(f"\nRunning {args.n_runs} experiments with different season splits")
    print(f"Test size: {args.test_size*100:.0f}% of seasons")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    if args.model == 'conversion':
        X, y, clean_data = load_conversion_data(df)
        target_col = 'converted'
    else:  # fg
        X, y, clean_data = load_fg_data(df)
        target_col = 'fg_made'
    
    print(f"✓ Loaded {len(X):,} samples with {X.shape[1]} features")
    
    # Get all available seasons
    all_seasons = sorted(clean_data['season'].unique())
    print(f"Available seasons: {all_seasons}")
    
    # Model parameters
    model_params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': 'binary:logistic',
        'random_state': args.random_seed,
        'n_jobs': -1
    }
    
    # Run experiments
    all_results = []
    
    for run_idx in range(args.n_runs):
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1}/{args.n_runs}")
        print("="*70)
        
        # Create random season split
        train_seasons, test_seasons = create_season_split(
            all_seasons, 
            args.test_size, 
            random_state=args.random_seed + run_idx
        )
        
        print(f"Train seasons: {sorted(train_seasons)}")
        print(f"Test seasons: {sorted(test_seasons)}")
        
        # Run experiment
        metrics, model = run_single_experiment(
            X, y, clean_data, 
            train_seasons, test_seasons,
            model_params, args.model
        )
        
        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
        print(f"  Brier:    {metrics['brier_score']:.4f}")
        print(f"  LogLoss:  {metrics['log_loss']:.4f}")
        
        all_results.append(metrics)
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Print summary
    print_results(aggregated)
    
    # Save results
    results = {
        'aggregated': aggregated,
        'individual_runs': all_results
    }
    
    save_results(results, args, args.output_dir)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()