"""
NFL 4th Down Decision Optimizer
Main script to load data and begin analysis
"""

import pandas as pd
import nfl_data_py as nfl

def load_nfl_data(years):
    """
    Load NFL play-by-play data for specified years
    
    Args:
        years: list of years to load (e.g., [2020, 2021, 2022])
    
    Returns:
        DataFrame with play-by-play data
    """
    print(f"Loading NFL data for years: {years}")
    
    # Load play-by-play data
    pbp_data = nfl.import_pbp_data(years)
    
    print(f"Loaded {len(pbp_data)} total plays")
    return pbp_data


def filter_fourth_downs(pbp_data):
    """
    Filter data to only 4th down plays
    
    Args:
        pbp_data: Full play-by-play DataFrame
    
    Returns:
        DataFrame with only 4th down plays
    """
    fourth_downs = pbp_data[pbp_data['down'] == 4].copy()
    
    print(f"Found {len(fourth_downs)} fourth down plays")
    
    # Show breakdown by play type
    if 'play_type' in fourth_downs.columns:
        print("\nPlay type breakdown:")
        print(fourth_downs['play_type'].value_counts())
    
    return fourth_downs


def explore_data(fourth_downs):
    """
    Basic exploration of 4th down data
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"\nColumns available: {len(fourth_downs.columns)}")
    print("\nKey columns:")
    key_cols = ['down', 'ydstogo', 'yardline_100', 'score_differential', 
                'qtr', 'play_type', 'fourth_down_converted', 'fourth_down_failed']
    
    for col in key_cols:
        if col in fourth_downs.columns:
            print(f"  - {col}")
    
    print("\nFirst few rows:")
    print(fourth_downs[['down', 'ydstogo', 'yardline_100', 'play_type', 
                        'desc']].head())
    
    print("\nBasic statistics:")
    print(fourth_downs[['ydstogo', 'yardline_100', 'score_differential']].describe())


def save_data(fourth_downs, filename='data/fourth_downs.csv'):
    """
    Save filtered data to CSV
    """
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    fourth_downs.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")


def main():
    """
    Main execution function
    """
    print("NFL 4th Down Optimizer - Data Loading")
    print("="*50)
    
    # Load data for recent seasons
    years = [2020, 2021, 2022, 2023]
    pbp_data = load_nfl_data(years)
    
    # Filter to 4th downs
    fourth_downs = filter_fourth_downs(pbp_data)
    
    # Explore the data
    explore_data(fourth_downs)
    
    # Save for later use
    save_data(fourth_downs)
    
    print("\n" + "="*50)
    print("Data loading complete!")
    print("="*50)


if __name__ == "__main__":
    main()