"""
Filter all play-by-play CSV files to only 4th down plays
Combines all years into single file
"""

import pandas as pd
import os
import glob
from datetime import datetime

def get_columns_to_keep():
    """
    Define which columns we actually need
    This drastically reduces file size
    """
    columns = [
        # Identifiers
        'play_id', 'game_id', 'season', 'week', 'game_date',

        #post or regular szn
        'season_type',
        
        # Teams
        'posteam', 'defteam', 'home_team', 'away_team', 'posteam_type',
        
        # Down and distance
        'down', 'ydstogo', 'yardline_100', 'goal_to_go',
        
        # Game situation
        'qtr', 'quarter_seconds_remaining', 'game_seconds_remaining',
        
        # Score
        'score_differential', 'posteam_score', 'defteam_score',
        
        # Timeouts
        'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
        
        # Play outcome
        'play_type', 'yards_gained', 'fourth_down_converted', 
        'fourth_down_failed', 'first_down',
        
        # Field goal
        'field_goal_result', 'kick_distance',
        
        # Advanced metrics
        'ep', 'epa', 'wp', 'wpa',
        
        # Play details
        'desc', 'shotgun', 'no_huddle',
        
        # Environment
        'roof', 'surface', 'temp', 'wind',
    ]
    
    return columns


def filter_single_file(filepath, columns_to_keep):
    """
    Load a single CSV and filter to 4th downs
    """
    try:
        print(f"  Loading {os.path.basename(filepath)}...", end=" ")
        
        # Read CSV, only load columns we need
        df = pd.read_csv(
            filepath,
            usecols=lambda x: x in columns_to_keep,
            low_memory=False
        )
        
        # Filter to 4th downs only
        fourth_downs = df[
                    (df['down'] == 4) & 
                    (df['season_type'] != 'PRE') &
                    (df['play_type'] != 'qb_kneel')
                ].copy()   
             
        print(f"✓ Found {len(fourth_downs)} 4th down plays")
        return fourth_downs
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    print("="*70)
    print("NFL 4TH DOWN DATA FILTER")
    print("="*70)
    
    # Find all CSV files in data folder
    csv_files = glob.glob("../data/raw_play_by_play/play_by_play_*.csv")
    csv_files.sort()  # Sort by year
    
    if not csv_files:
        print("\n❌ No CSV files found in data/ folder!")
        print("Make sure your files are named: play_by_play_YYYY.csv")
        return
    
    print(f"\nFound {len(csv_files)} CSV files")
    print(f"Years: {os.path.basename(csv_files[0])} to {os.path.basename(csv_files[-1])}")
    
    # Define columns to keep
    columns_to_keep = get_columns_to_keep()
    print(f"\nKeeping {len(columns_to_keep)} columns (out of ~370)")
    
    # Process each file
    print(f"\nProcessing files...")
    all_fourth_downs = []
    
    start_time = datetime.now()
    
    for filepath in csv_files:
        fourth_downs = filter_single_file(filepath, columns_to_keep)
        
        if fourth_downs is not None and len(fourth_downs) > 0:
            all_fourth_downs.append(fourth_downs)
    
    # Combine all years
    print(f"\nCombining all years...")
    combined = pd.concat(all_fourth_downs, ignore_index=True)
    
    # Statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total 4th down plays: {len(combined):,}")
    print(f"Seasons: {combined['season'].min()} - {combined['season'].max()}")
    print(f"Columns: {len(combined.columns)}")
    
    # Play type breakdown
    print(f"\nPlay type breakdown:")
    print(combined['play_type'].value_counts())
    
    # Conversion stats (for plays where they went for it)
    went_for_it = combined[combined['play_type'].isin(['pass', 'run'])]
    if len(went_for_it) > 0:
        converted = went_for_it['fourth_down_converted'].sum()
        conversion_rate = (converted / len(went_for_it)) * 100
        print(f"\nWent for it: {len(went_for_it):,} times")
        print(f"Converted: {converted:,} ({conversion_rate:.1f}%)")
    
    # Save to file
    output_file = "../data/filtered_data/fourth_downs_filtered.csv"
    print(f"\nSaving to {output_file}...")
    combined.to_csv(output_file, index=False)
    
    # File size info
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    elapsed = datetime.now() - start_time
    
    print(f"✓ Saved! ({file_size_mb:.1f} MB)")
    print(f"✓ Processing time: {elapsed.total_seconds():.1f} seconds")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nFiltered data saved to: {output_file}")
    print("Next step: Feature engineering and model training")


if __name__ == "__main__":
    main()