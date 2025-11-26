"""
Stage 2: Fourth Down Play Type Outcome Prediction
Predicts success probability for different play types (go for it, FG, punt)
Uses XGBoost for optimal play type recommendation
"""

import argparse
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


############################################
# Argument Parsing
############################################
"""
Parse command line arguments for XGBoost hyperparameters
Arguments:
    None
Returns:
    args -- Parsed arguments namespace
"""
def parse_args():
    #XGBoost hyperparameters
    parser = argparse.ArgumentParser(description='Stage 2: Fourth Down Play Type Outcome Prediction using XGBoost')

    parser.add_argument('--data-path',  type=str,  default='data/filtered_data/fourth_downs_with_features.csv', help='Path to the CSV data file')
    
    parser.add_argument('--output-dir',  type=str,  default='models', help='Directory to save models') 

    # Model selection
    parser.add_argument('--model', type=str, default='all',choices=['conversion', 'fg', 'all'], help='Which model to train')

    # Train/test split
    parser.add_argument('--test-seasons', nargs='+', type=int,default=[2023, 2024], help='Seasons to use for testing')
    
    # XGBoost hyperparameters
    parser.add_argument('--max-depth', type=int, default=6, help='Maximum tree depth (default: 6)')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Learning rate / eta (default: 0.05)')    
    parser.add_argument('--n-estimators', type=int, default=1000, help='Number of boosting rounds (default: 1000)')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio of training instances (default: 0.8)')
    parser.add_argument('--colsample-bytree', type=float, default=0.8, help='Subsample ratio of columns when constructing each tree (default: 0.8)')
    parser.add_argument('--min-child-weight', type=int, default=5, help='Minimum sum of instance weight needed in a child (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Minimum loss reduction required to make a split (default: 0.1)')
    parser.add_argument('--reg-alpha', type=float, default=0.5, help='L1 regularization term on weights (default: 0.5)')
    parser.add_argument('--reg-lambda', type=float, default=1.0, help='L2 regularization term on weights (default: 1.0)')
    parser.add_argument('--scale-pos-weight', type=float, default=None, help='Balance of positive and negative weights (auto-calculated if None)')
    parser.add_argument('--early-stopping-rounds', type=int, default=50, help='Early stopping rounds (default: 50)')

    # Other options
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress information')
    
    return parser.parse_args()


############################################
# Load Data
############################################
"""
Load the processed data 
Args:
    filepath (str): Path to CSV file
Returns:
    pd.DataFrame: Loaded data
"""
def load_data(filepath):
    
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} fourth down plays\n")
    
    return df


############################################
# CONVERSION MODEL
############################################
'''
Conversion Model using XGBoost
'''
class ConversionModelXGB:


    ############################
    # Model Initialization
    ############################

    def __init__(self, args):
        self.params = {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_child_weight': args.min_child_weight,
            'gamma': args.gamma,
            'reg_alpha': args.reg_alpha,
            'reg_lambda': args.reg_lambda,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': args.random_state,
            'use_label_encoder': False
        }

        self.model = xgb.XGBClassifier(**self.params)
        self.features = None
        self.early_stopping_rounds = args.early_stopping_rounds


    ############################
    #Prepare Data
    ############################
    """
    Prepare data for conversion model
    Only include plays where team went for it
        
    Args:
        df (pd.DataFrame): Full data
        
    Returns:
        tuple: (X, y, clean_data)
    """
    def prepare_data(self, df):

        print("="*70)
        print("PREPARING CONVERSION MODEL DATA USING XGBOOST")
        print("="*70)

        #filter for just fourth down plays where teams went for it 
        went_for_it = df[df['went_for_it'] == 1].copy()

        print(f"\nTotal 4th downs: {len(df):,}")
        print(f"Went for it: {len(went_for_it):,} ({len(went_for_it)/len(df)*100:.1f}%)")

        #features to use
        self.features = [
            'yards_to_go',
            'field_position',
            'score_diff',
            'time_remaining',
            'quarter',
            'is_home',
            'in_red_zone',
            'in_fg_range',
            'in_own_territory',
            'short_yardage',
            'very_short',
            'long_distance',
            'losing',
            'winning',
            'tied',
            'close_game',
            'late_game',
            'final_2_min',
            'desperate_situation',
            'is_dome',
            'is_outdoors',
            'cold_weather',
            'windy',
        ]

        available_features = [ f for f in self.features if f in went_for_it.columns]
        self.features = available_features

        print(f"Using {len(self.features)} features")

        clean_data = went_for_it[self.features + ['converted', 'season']].dropna()

        print(f"Cleaned data has {len(clean_data):,} plays after dropping missing values\n")
        
        X = clean_data[self.features]
        y = clean_data['converted']

        return X, y, clean_data
    

    ############################
    # Train Model
    ############################
    """
    Train XGBoost model       
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
    """
    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("\n" + "="*70)
        print("TRAINING CONVERSION MODEL (XGBoost)")
        print("="*70)

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
            print(f"Model Trained (stopped at {self.model.best_iteration} iterations)\n")
        else:
            self.model.fit(X_train, y_train)
            print("Model Trained\n")

        print("\nFeatures ranked:")
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(importance_df.head().to_string(index=False))


    ############################
    # Evaluate Model
    ############################
    """
    Evaluate model    
    Args:
        X_test: Test features
        y_test: Test labels    
    Returns:
        dict: Evaluation metrics
    """
    def evaluate(self, X_test, y_test):
        print("\n" + "="*70)
        print("CONVERSION MODEL EVALUATION")
        print("="*70)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"AUC-ROC: {auc:.3f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(
            cm, 
            index=['True Failed', 'True Converted'],
            columns=['Predicted Failed', 'Predicted Converted']
        ))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Failed', 'Converted']))
        
        return {'accuracy': accuracy, 'auc': auc}
    


    ############################
    # Predict probability
    ############################
    '''
    Args:
        X (pd.DataFrame): Features to predict
    Returns:
        np.ndarray: Predicted probabilities
    '''
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    

    ############################
    # Save Model
    ############################
    def save_model(self, filepath='models/stage2_conversion_xgb.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)    
        print(f"\nModel saved to {filepath}")


############################################
# Field Goal Model using XGBoost
#############################################
class FieldGoalModelXGB:
    #predicts FG success probability using XGBoost model

    def __init__(self, args):
        """Initialize XGBoost FG model"""
        self.params = {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'min_child_weight': args.min_child_weight,
            'gamma': args.gamma,
            'reg_alpha': args.reg_alpha,
            'reg_lambda': args.reg_lambda,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': args.random_state,
            'use_label_encoder': False
        }

        self.model = xgb.XGBClassifier(**self.params)
        self.features = None
        self.early_stopping_rounds = args.early_stopping_rounds

    
    ############################
    #Prepare Data
    ############################
    '''
    Prepare data for field goal model
    Only include plays where team attempted a field goal
    Args:
        df (pd.DataFrame): Full data
    '''
    def prepare_data(self, df):
        print("="*70)
        print("PREPARING FIELD GOAL MODEL DATA (XGBoost)")
        print("="*70)

        fg_attempts = df[df['attempted_field_goal'] == 1].copy()

        print(f"\nTotal 4th downs: {len(df):,}")
        print(f"FG attempts: {len(fg_attempts):,} ({len(fg_attempts)/len(df)*100:.1f}%)")

        #caluclate fg distance if needed
        if 'fg_distance' not in fg_attempts.columns:
            fg_attempts['fg_distance'] = fg_attempts['field_poition'] + 17

        self.features = [
            'fg_distance',
            'is_home',
            'quarter',
            'score_diff',
            'is_dome',
            'is_outdoors',
            'cold_weather',
            'windy',
            'late_game',
            'final_2_min',
        ]

        available_features = [ f for f in self.features if f in fg_attempts.columns]
        self.features = available_features

        print(f"Using {len(self.features)} features")

        clean_data = fg_attempts[self.features + ['fg_made', 'season']].dropna()
        print(f"Cleaned data has {len(clean_data):,} plays after dropping missing values\n")

        X = clean_data[self.features]
        y = clean_data['fg_made']

        return X, y, clean_data

    ############################
    # Train Model
    ############################
    '''
    Train XGBoost model
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
    '''  
    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("="*70)
        print("TRAINING FIELD GOAL MODEL (XGBoost)")
        print("="*70)

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
            print(f"Model Trained (stopped at {self.model.best_iteration} iterations)\n")
        else:
            self.model.fit(X_train, y_train)
            print("Model Trained\n")

        print("\nFeatures ranked:")
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(importance_df.head().to_string(index=False))

    

    ############################
    # Evaluate Model
    ############################
    '''
    Evaluate model    
    Args:
        X_test: Testing features
        y_test: Testing labels
    '''
    def evaluate(self, X_test, y_test):
        print("\n" + "="*70)
        print("FIELD GOAL MODEL EVALUATION")
        print("="*70)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nAccuracy: {accuracy:.3f}")
        print(f"AUC_ROC: {auc:.3f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(
            cm,
            index=['True Missed', 'True Made'],
            columns=['Predicted Missed', 'Predicted Made']
        ))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Missed', 'Made']))

        return {'accuracy': accuracy, 'auc': auc}
    

    ############################
    # Predict probability
    ############################
    '''    
    Args:
        X (pd.DataFrame): Features to predict
    Returns:
        np.ndarray: Predicted probabilities
    '''    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    

    ############################
    # Save Model
    ############################
    def save(self, filepath='models/stage2_fieldgoal_xgb.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)   
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)    
        print(f"\n✓ Model saved to {filepath}")


############################################
# Split by Season
#############################################

def split_by_season(df, X, y, test_seasons):
    print(f"\n{'='*70}")
    print("SPLITTING DATA BY SEASON")
    print("="*70)

    if 'season' not in df.columns:
        raise ValueError("Dataframe must contain 'season' column for season-based splitting")

    train_mask = ~df['season'].isin(test_seasons)
    test_mask = df['season'].isin(test_seasons)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Train: {len(X_train):,} plays")
    print(f"Test: {len(X_test):,} plays")

    return X_train, X_test, y_train, y_test


############################################
# Main Execution
############################################
def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("STAGE 2: OUTCOME PREDICTION (XGBoost)")
    print("="*70)
    
    if not args.quiet:
        print(f"\nConfiguration:")
        print(f"  Max depth: {args.max_depth}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  N estimators: {args.n_estimators}")
        print(f"  Test seasons: {args.test_seasons}")
        print(f"  Training: {args.model}")

if __name__ == "__main__":
    main()