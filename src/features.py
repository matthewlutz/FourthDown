"""
Feature engineering for 4th down decision optimization
Takes filtered 4th down data and creates all features needed for modeling
"""

import pandas as pd
import numpy as np
import os


def load_filtered_data():
    """Load filtered fourth down data"""
    filepath = "data/filtered_data/fourth_downs_filtered.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Filtered data not found at {filepath}\n"
            "Run filter_data.py first"
        )
    
    print(f"Loading filtered data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} fourth down plays")
    return df


def create_features(df):
    """Create all features from existing columns"""
    
    print("\nCreating features...")
    df = df.copy()  # Make copy to avoid modifying original
    
    # ============= FIELD POSITION =============
    df['field_position'] = df['yardline_100']
    df['yards_to_go'] = df['ydstogo']



    # ============= SCORE DIFFERENTIAL =============
    df['score_diff'] = df['score_differential']



    # ============= TIME REMAINING =============
    if 'game_seconds_remaining' in df.columns:
        df['time_remaining'] = df['game_seconds_remaining']
    else:
        df['time_remaining'] = (4 - df['qtr']) * 900 + df.get('quarter_seconds_remaining', 450)
    df['quarter'] = df['qtr']



    # ============= HOME/AWAY =============
    df['is_home'] = (df['posteam_type'] == 'home').astype(int)



    # ============= FIELD ZONES =============
    df['in_red_zone'] = (df['field_position'] <= 20).astype(int)
    df['in_fg_range'] = (df['field_position'] <= 35).astype(int)
    df['in_own_territory'] = (df['field_position'] > 50).astype(int)
    df['at_midfield'] = ((df['field_position'] >= 40) & (df['field_position'] <= 60)).astype(int)



    # ============= DISTANCE FEATURES =============
    df['short_yardage'] = (df['yards_to_go'] <= 2).astype(int)
    df['very_short'] = (df['yards_to_go'] == 1).astype(int)
    df['long_distance'] = (df['yards_to_go'] >= 5).astype(int)
    df['fourth_and_inches'] = (df['yards_to_go'] < 1).astype(int)



    # ============= SCORE SITUATION =============
    df['losing'] = (df['score_diff'] < 0).astype(int)
    df['winning'] = (df['score_diff'] > 0).astype(int)
    df['tied'] = (df['score_diff'] == 0).astype(int)
    df['close_game'] = (abs(df['score_diff']) <= 8).astype(int)
    df['blowout'] = (abs(df['score_diff']) > 16).astype(int)



    # ============= TIME PRESSURE =============
    df['late_game'] = (df['time_remaining'] < 300).astype(int)
    df['final_2_min'] = (df['time_remaining'] < 120).astype(int)
    df['very_late'] = (df['time_remaining'] < 60).astype(int)



    # ============= CRITICAL SITUATIONS =============
    df['desperate_situation'] = ((df['losing'] == 1) & (df['late_game'] == 1)).astype(int)
    df['protect_lead'] = ((df['winning'] == 1) & (df['late_game'] == 1)).astype(int)



    # ============= GAME TYPE =============
    df['is_playoff'] = (df['season_type'] == 'POST').astype(int)



    # ============= INTERACTION FEATURES =============
    df['distance_field_interaction'] = df['yards_to_go'] * df['field_position']
    df['score_time_pressure'] = df['score_diff'] * (3600 - df['time_remaining'])
    df['short_yardage_red_zone'] = df['short_yardage'] * df['in_red_zone']
    df['long_distance_own_territory'] = df['long_distance'] * df['in_own_territory']



    # ============= TARGET VARIABLES =============
    df['went_for_it'] = df['play_type'].isin(['pass', 'run']).astype(int)
    if 'fourth_down_converted' in df.columns:
        df['converted'] = df['fourth_down_converted'].fillna(0).astype(int)
    else:
        df['converted'] = (df['yards_gained'] >= df['yards_to_go']).astype(int)



    # ============= PLAY TYPE FLAGS =============
    df['is_punt'] = (df['play_type'] == 'punt').astype(int)
    df['is_field_goal'] = (df['play_type'] == 'field_goal').astype(int)
    df['is_pass'] = (df['play_type'] == 'pass').astype(int)
    df['is_run'] = (df['play_type'] == 'run').astype(int)



    # ============= FIELD GOAL FEATURES =============
    if 'field_goal_result' in df.columns:
        df['fg_made'] = (df['field_goal_result'] == 'made').astype(int)
        df['fg_missed'] = (df['field_goal_result'] == 'missed').astype(int)
        df['fg_blocked'] = (df['field_goal_result'] == 'blocked').astype(int)
    else:
        df['fg_made'] = 0
        df['fg_missed'] = 0
        df['fg_blocked'] = 0
    if 'kick_distance' in df.columns:
        df['fg_distance'] = df['kick_distance']
    else:
        df['fg_distance'] = df['field_position'] + 17



    # ============= STADIUM FEATURES =============
    if 'roof' in df.columns:
        df['is_dome'] = (df['roof'] == 'dome').astype(int)
        df['is_outdoors'] = (df['roof'] == 'outdoors').astype(int)
        df['is_retractable'] = (df['roof'].isin(['retractable', 'open'])).astype(int)
    else:
        df['is_dome'] = 0
        df['is_outdoors'] = 1
        df['is_retractable'] = 0



    # ============= WEATHER FEATURES =============
    if 'temp' in df.columns:
        df['cold_weather'] = ((df['temp'] < 40) & (df['is_outdoors'] == 1)).astype(int)
        df['hot_weather'] = ((df['temp'] > 85) & (df['is_outdoors'] == 1)).astype(int)
    else:
        df['cold_weather'] = 0
        df['hot_weather'] = 0
    if 'wind' in df.columns:
        df['windy'] = ((df['wind'] > 15) & (df['is_outdoors'] == 1)).astype(int)
    else:
        df['windy'] = 0
    
    print(f"✓ Created features")
    return df


def clean_data(df):
    """Clean data and handle missing values"""
    print("\nCleaning data...")
    
    before = len(df)
    critical_cols = ['yards_to_go', 'field_position', 'play_type']
    df = df.dropna(subset=critical_cols).copy()
    after = len(df)
    dropped = before - after
    
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} rows with missing critical values")
    
    # Fill missing values
    df['score_diff'] = df['score_diff'].fillna(0)
    df['time_remaining'] = df['time_remaining'].fillna(1800)
    df['quarter'] = df['quarter'].fillna(2)
    df['is_home'] = df['is_home'].fillna(0)
    
    print(f"✓ Cleaned data: {len(df):,} rows ready")
    return df


def get_feature_list():
    """Return list of all engineered features for modeling"""
    features = [
        'yards_to_go', 'field_position', 'score_diff', 'time_remaining', 'quarter', 'is_home',
        'in_red_zone', 'in_fg_range', 'in_own_territory', 'at_midfield',
        'short_yardage', 'very_short', 'long_distance', 'fourth_and_inches',
        'losing', 'winning', 'tied', 'close_game', 'blowout',
        'late_game', 'final_2_min', 'very_late',
        'desperate_situation', 'protect_lead', 'is_playoff',
        'distance_field_interaction', 'score_time_pressure', 'short_yardage_red_zone', 'long_distance_own_territory',
        'is_dome', 'is_outdoors', 'is_retractable', 'cold_weather', 'hot_weather', 'windy',
        'fg_distance',
    ]
    return features


def main():
    """Main feature engineering pipeline"""
    print("="*70)
    print("NFL 4TH DOWN - FEATURE ENGINEERING")
    print("="*70)
    
    df = load_filtered_data()
    df = create_features(df)
    df = clean_data(df)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    features = get_feature_list()
    print(f"Engineered features: {len(features)}")
    
    print("\nTarget variable distribution:")
    print(f"  Went for it: {df['went_for_it'].sum():,} ({df['went_for_it'].mean()*100:.1f}%)")
    print(f"  Punted: {df['is_punt'].sum():,} ({df['is_punt'].mean()*100:.1f}%)")
    print(f"  Field goal: {df['is_field_goal'].sum():,} ({df['is_field_goal'].mean()*100:.1f}%)")
    
    went_for_it = df[df['went_for_it'] == 1]
    if len(went_for_it) > 0:
        conv_rate = went_for_it['converted'].mean()
        print(f"\nConversion rate (when going for it): {conv_rate*100:.1f}%")
    
    output_file = "data/filtered_data/fourth_downs_with_features.csv"
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved! ({file_size_mb:.1f} MB)")
    
    print("\n" + "="*70)
    print("Next step: Train models with main.py")
    print("="*70)


if __name__ == "__main__":
    main()