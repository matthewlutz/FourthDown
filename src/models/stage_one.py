"""
Stage 1: Play Type Prediction Model
Predicts what type of play will be called on 4th down:
- pass
- run  
- field_goal
- punt
"""

from py_compile import main
import pandas as pd
import numpy as np      
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse


###################################################################
# Argument Parsing for Model Configuration
###################################################################

def parse_arguments():
    """
    Parse command-line arguments for model configuration
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Stage 1: NFL 4th Down Play Type Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Inverse regularization strength (lower = more regularization)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=2000,
        help='Maximum iterations for model convergence'
    )
    
    # Cross-validation settings
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation (faster training)'
    )
    
    # Train/test split settings
    parser.add_argument(
        '--test-seasons',
        nargs='+',
        type=int,
        default=[2023, 2024],
        help='Seasons to use for testing (e.g., --test-seasons 2023 2024)'
    )
    
    # Data settings
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/filtered_data/fourth_downs_with_features.csv',
        help='Path to input data CSV file'
    )
    
    # Output settings
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/stage1_play_type_logreg.pkl',
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )
    
    # Verbosity
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser.parse_args()


###################################################################
#this function loads data with features for modeling        
####################################################################
"""
Args:       - filepath (str): Path to the CSV file containing fourth down data with features   
Returns:    - pd.DataFrame: Loaded DataFrame with all fourth down plays and features
"""
def load_data(filepath):
    """
    Load the processed data for modeling
    """
    print(f"Loading data from {filepath}...")
    filepath = "data/filtered_data/fourth_downs_with_features.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data not found at {filepath}\n"
            "src/features.py first"
        )

    df = pd.read_csv(filepath)
    print(f"Data loaded successfully from {filepath}")
    return df



###################################################################
#this function prepares data for play type prediction
###################################################################

"""
Args:       -df_dirty (pd.DataFrame): Raw DataFrame with all plays and columns

Returns:    - tuple: (X, y, available_features, df)
            - X (pd.DataFrame): Feature matrix for modeling
            - y (pd.Series): Target variable (play_type)
            - available_features (list): List of feature column names
            - df (pd.DataFrame): Cleaned DataFrame with only valid plays
"""
def prepare_play_type_data(df_dirty):
    """
    Prepare data for play type prediction
    We want to predict: pass, run, field_goal, or punt
    """
    print("=" * 70)
    print("Preparing data for play type prediction...")
    print("=" * 70)

    valid_play_types = ['pass', 'run', 'field_goal', 'punt'] 
    df = df_dirty[df_dirty['play_type'].isin(valid_play_types)].copy() #we only want regular plays from scrimmage, no kickoffs

    print(f"\nPlay type distribution:")
    print(df['play_type'].value_counts())
    print(f"\nTotal plays: {len(df):,}")

    #features we will use to predict play type
    features = [
        # Basic situation
        'yards_to_go',
        'field_position',
        'score_diff',
        'time_remaining',
        'quarter',
        'is_home',
        
        # Field zones
        'in_red_zone',
        'in_fg_range',
        'in_own_territory',
        'at_midfield',
        
        # Distance categories
        'short_yardage',
        'very_short',
        'long_distance',
        'fourth_and_inches',
        
        # Score situation
        'losing',
        'winning',
        'tied',
        'close_game',
        'blowout',
        
        # Time pressure
        'late_game',
        'final_2_min',
        'very_late',
        
        # Critical situations
        'desperate_situation',
        'protect_lead',
        
        # Game type
        'is_playoff',
        
        # Interactions
        'distance_field_interaction',
        'score_time_pressure',
        'short_yardage_red_zone',
        'long_distance_own_territory',
        
        # Environment
        'is_dome',
        'is_outdoors',
        'cold_weather',
        'hot_weather',
        'windy',
    ]
    
    available_features = [f for f in features if f in df.columns]
    print(f"\nUsing {len(available_features)} features") 

    # Drop rows with missing features
    df = df[available_features + ['play_type', 'season']].dropna()
    X = df[available_features]
    y = df['play_type']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print(f"Final dataset: {len(df):,} plays")

    return X, y, available_features, df



###################################################################
#this function splits data by season for proper time-based splitting
###################################################################

"""
Args:       - df (pd.DataFrame): DataFrame containing play data with a 'season' column
            - X (pd.DataFrame): Feature matrix
            - y (pd.Series): Target variable
            - test_seasons (list): List of seasons to use for the test set
Returns:    tuple: (X_train, X_test, y_train, y_test)
            - X_train (pd.DataFrame): Training features
            - X_test (pd.DataFrame): Testing features
            - y_train (pd.Series): Training labels
            - y_test (pd.Series): Testing labels
"""
def split_by_season(df, X, y, test_seasons=[]):
    """
    Split data by season (proper time-based split)
    """
    print(f"\n{'='*70}")
    print("SPLITTING DATA BY SEASON")
    print("="*70)


    if 'season' not in df.columns:
        raise ValueError("DataFrame must contain 'season' column for season-based splitting")

    #masks for train/test split
    train_mask = ~df['season'].isin(test_seasons)
    test_mask = df['season'].isin(test_seasons)

    X_train = X[train_mask]
    X_test = X[test_mask]       
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Print split info
    print(f"Train seasons: {sorted(df[train_mask]['season'].unique())}")
    print(f"Test seasons: {test_seasons}")
    print(f"\nTrain set: {len(X_train):,} plays")
    print(f"Test set: {len(X_test):,} plays")
    
    print("\nTrain set distribution:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest set distribution:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test



###################################################################
# Cross-validation for robust model evaluation
###################################################################
"""
Args:       - X (pd.DataFrame): Feature matrix
            - y (pd.Series): Target variable
            - feature_names (list): List of feature column names
            - cv (int): Number of cross-validation folds
Returns:    - dict: Dictionary with CV results (mean, std, all scores)
"""
def cross_validate_model(X, y, feature_names, cv=5, C=1.0, max_iter=2000):
    """
    Evaluate model using k-fold cross-validation
    Gives more robust estimate of model performance
    """
    from sklearn.model_selection import cross_val_score, cross_validate
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION ({cv}-Fold)")
    print("="*70)
    
    # Create model
    model = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        C=C,
        max_iter=max_iter
    )
    
    # Run cross-validation
    print(f"\nRunning {cv}-fold cross-validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Print results for each fold
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    
    print(f"\n{'='*50}")
    print(f"Mean Accuracy: {scores.mean():.3f}")
    print(f"Std Deviation: {scores.std():.3f}")
    print(f"95% Confidence Interval: [{scores.mean() - 2*scores.std():.3f}, {scores.mean() + 2*scores.std():.3f}]")
    print("="*50)
    
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'all_scores': scores,
        'feature_count': len(feature_names)
    }


####################################################################
## Logistic Regression Model for Play Type Prediction
####################################################################

"""
    Logistic Regression model for play type prediction
    Multi-class classification (4 play types: pass, run, field_goal, punt)
 """
class PlayTypeModel:

    ###################################
    # INIT MODEL
    ###################################
    """
        Initialize the logistic regression model
        
        Args:
            C (float): Inverse regularization strength (lower = more regularization)
                      Default: 1.0
    """
    def __init__(self, C=1.0, max_iter=2000):

        self.model = LogisticRegression(C=C, max_iter=max_iter, multi_class='multinomial', solver='lbfgs')

        ##################################
        '''
        Parameters for log reg model:
        - C: Inverse regularization strength
        - max_iter: Maximum number of iterations
        - multi_class: Type of multi-class classification
        - solver: Algorithm to use for optimization
        '''
        self.model = LogisticRegression(
            solver='lbfgs', #algorithm for optimization
            penalty='l2', #L2 regularization
            C=C, #inverse regularization strength
            max_iter=max_iter, #max iterations for convergence
            #random_state=42 #for reproducibility
        )
        self.features = None


    ###################################
    # TRAIN MODEL
    ###################################
    """
        Train the logistic regression model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels (play types)
            feature_names (list): List of feature column names
        
        Returns:
            None
    """
    def train(self, X_train, y_train, feature_names):        
        print("\n" + "="*70)
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print("="*70)
        
        self.features = feature_names
        self.model.fit(X_train, y_train)
        
        print("✓ Model trained")
        
        # insight
        print("\nTop 10 features by importance (averaged across classes):")
        coef_df = pd.DataFrame(
            self.model.coef_,
            columns=feature_names,
            index=self.model.classes_
        )
        avg_importance = coef_df.abs().mean(axis=0).sort_values(ascending=False)
        print(avg_importance.head(10))


    ###################################
    # EVALUATE MODEL
    # ###################################
    """
        Evaluate model performance on test set
        
        Args:
            X_test (pd.DataFrame): Testing features
            y_test (pd.Series): Testing labels (play types)
        
        Returns:
            float: Overall accuracy score
    """
    def evaluate(self, X_test, y_test):
        print("\n" + "="*70)
        print("EVALUATION - LOGISTIC REGRESSION")
        print("="*70)

        #predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.3f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {c}" for c in self.model.classes_],
            columns=[f"Pred {c}" for c in self.model.classes_]
        )
        print(cm_df)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.model.classes_))
        
        return accuracy
    


    ###################################
    #Predict Probabilities
    ################################### 
    """
        Predict probabilities for each play type
        
        Args:
            X (pd.DataFrame or np.array): Features for prediction
        
        Returns:
                np.array: Probability matrix [n_samples, n_classes]
                Columns correspond to model.classes_ order
    """
    def predict_probability(self, X):
        return self.model.predict_probability(X)
    


    ###################################
    #PREDICT PLAY TYPE
    ################################### 
    """
        Predict play types
        
        Args:
            X (pd.DataFrame or np.array): Features for prediction
        
        Returns:
            np.array: Predicted play types
    """
    def predict(self, X):
        return self.model.predict(X)


    ############################################
    """
        Save trained model to disk using pickle
        Args:
            filepath (str): Path where model will be saved
                    Default: 'models/stage1_play_type_logreg.pkl'
        
        Returns:
            None
    """
    def save(self, filepath='models/stage1_play_type_logreg.pkl'):
        
        os.makedirs('models', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"\n✓ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/stage1_play_type_logreg.pkl'):
        """
        Load a saved model from disk
        
        Args:
            filepath (str): Path to saved model file
                           Default: 'models/stage1_play_type_logreg.pkl'
        
        Returns:
            PlayTypePredictorLogReg: Loaded model instance
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

############################################
# Main training pipeline for Stage 1 play type prediction
#############################################
"""
    Main training pipeline for Stage 1 play type prediction     
    Args: None
        
    Returns: None
"""
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    if not args.quiet:
        print("\n" + "="*70)
        print("STAGE 1: PLAY TYPE PREDICTION")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  C (regularization): {args.C}")
        print(f"  Max iterations: {args.max_iter}")
        print(f"  CV folds: {args.cv_folds}")
        print(f"  Test seasons: {args.test_seasons}")
        print(f"  Data path: {args.data_path}")
    
    # Load data
    df = load_data(args.data_path)
    
    # Prepare data
    X, y, feature_names, df_clean = prepare_play_type_data(df)
    
    # Split by season
    X_train, X_test, y_train, y_test = split_by_season(
        df_clean, X, y, test_seasons=args.test_seasons
    )
    
    # Train model
    model = PlayTypeModel(C=args.C, max_iter=args.max_iter)
    model.train(X_train, y_train, feature_names)
    
    # Evaluate
    accuracy = model.evaluate(X_test, y_test)
    
    # Save model (unless --no-save)
    if not args.no_save:
        model.save(args.model_path)
    
    # Cross-validation (unless --no-cv)
    if not args.no_cv:
        cv_results = cross_validate_model(
            X, y, feature_names, 
            cv=args.cv_folds, 
            C=args.C, 
            max_iter=args.max_iter
        )
        
        print("\n" + "="*70)
        print("STAGE 1 COMPLETE")
        print("="*70)
        print(f"Single Split Accuracy: {accuracy:.3f}")
        print(f"CV Mean Accuracy:      {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    else:
        print("\n" + "="*70)
        print("STAGE 1 COMPLETE")
        print("="*70)
        print(f"Model Accuracy: {accuracy:.3f}")
    
    print("\nNext step: Build Stage 2 - Outcome Prediction Models")


if __name__ == "__main__":
    main()


