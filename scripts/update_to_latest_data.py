"""
Update the dataset with the most recent 2024 and 2025 season data.

This script:
1. Collects fresh data from FanGraphs for 2021-2025
2. Re-engineers all 59 features
3. Creates next_season_strikeouts target
4. Replaces the existing dataset
"""

import pandas as pd
import numpy as np
from pybaseball import pitching_stats
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Engineer all 59 features from raw FanGraphs data."""
    
    print("  Engineering features...")
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'Name': 'full_name',
        'Season': 'season',
        'IP': 'total_innings_pitched',
        'SO': 'total_strikeouts',
        'G': 'games_pitched',
        'K/9': 'k_per_9',
        'BB/9': 'bb_per_9',
        'HR/9': 'hr_per_9',
        'H/9': 'h_per_9',
        'BB': 'total_walks',
        'K/BB': 'k_bb_ratio',
        'K%': 'strike_percentage',
        'ERA': 'season_era',
        'WHIP': 'season_whip',
        'FIP': 'fip',
        'xFIP': 'xFIP',
        'SIERA': 'SIERA',
        'SwStr%': 'swstr_pct',
        'CSW%': 'CSW%',
        'Contact%': 'Contact%',
        'O-Swing%': 'O-Swing%',
        'Z-Swing%': 'Z-Swing%',
        'Zone%': 'Zone%',
        'F-Strike%': 'F-Strike%',
        'Hard%': 'Hard%',
        'Barrel%': 'Barrel%',
        'AVG': 'batting_avg_against',
        'LOB%': 'lob_pct',
        'Age': 'age',
        'Stuff+': 'stuff_plus',
        'Location+': 'command_plus',
        'Pitches': 'total_pitches',
    })
    
    # Age features
    df['age_squared'] = df['age'] ** 2
    df['is_prime_age'] = ((df['age'] >= 27) & (df['age'] <= 31)).astype(int)
    df['is_young'] = (df['age'] < 27).astype(int)
    df['is_veteran'] = (df['age'] > 31).astype(int)
    df['age_from_peak'] = abs(df['age'] - 29)
    
    # Fill missing Stuff+/Command+ with 100 (league average)
    df['stuff_plus'] = df['stuff_plus'].fillna(100)
    df['command_plus'] = df['command_plus'].fillna(100)
    
    # Engineered features
    df['k_minus_bb_pct'] = df['strike_percentage'] - (df['total_walks'] / (df['total_innings_pitched'] * 3) * 100)
    df['contact_quality'] = df['Hard%'] + df['Barrel%']
    df['whiff_rate'] = df['swstr_pct']
    df['zone_contact_diff'] = 100 - df['Contact%']
    df['true_outcomes_pct'] = (df['total_strikeouts'] + df['total_walks'] + (df['hr_per_9'] * df['total_innings_pitched'] / 9)) / (df['total_innings_pitched'] * 3) * 100
    df['k_to_contact_ratio'] = df['total_strikeouts'] / (df['total_innings_pitched'] * 3 - df['total_strikeouts'])
    df['k_upside'] = df['k_per_9'] * df['swstr_pct'] / 100
    df['pitch_efficiency'] = df['total_innings_pitched'] / df['total_pitches'] * 100
    df['power_index'] = df['k_per_9'] * (1 - df['Hard%'] / 100)
    df['consistency_score'] = 1 / (df['season_whip'] * df['season_era'])
    
    # Role indicators
    df['is_ace'] = ((df['k_per_9'] >= 9.5) & (df['total_innings_pitched'] >= 150)).astype(int)
    df['is_high_k_pitcher'] = (df['k_per_9'] >= 10.0).astype(int)
    df['is_workhorse'] = (df['total_innings_pitched'] >= 180).astype(int)
    df['log_total_strikeouts'] = np.log1p(df['total_strikeouts'])
    
    # Binary roles (starter vs reliever)
    # Starters typically: GS > 15 OR IP/G > 4
    df['games_started'] = df.get('GS', 0)
    df['is_starter'] = ((df['games_started'] > 15) | ((df['total_innings_pitched'] / df['games_pitched']) > 4)).astype(int)
    df['is_reliever'] = (df['is_starter'] == 0).astype(int)
    
    # Interactions
    df['k9_x_starter'] = df['k_per_9'] * df['is_starter']
    df['swstr_x_starter'] = df['swstr_pct'] * df['is_starter']
    df['ip_x_starter'] = df['total_innings_pitched'] * df['is_starter']
    df['swstr_x_ip'] = df['swstr_pct'] * df['total_innings_pitched']
    df['k9_x_era'] = df['k_per_9'] * df['season_era']
    df['age_x_ip'] = df['age'] * df['total_innings_pitched']
    df['stuff_x_command'] = df['stuff_plus'] * df['command_plus']
    df['workload_stress'] = df['total_pitches'] / df['age']
    
    return df

def create_target(df):
    """Create next_season_strikeouts target."""
    print("  Creating target variable...")
    
    df = df.sort_values(['full_name', 'season'])
    df['next_season_strikeouts'] = df.groupby('full_name')['total_strikeouts'].shift(-1)
    
    return df

def main():
    print("=" * 70)
    print("UPDATING DATASET WITH LATEST 2024-2025 DATA")
    print("=" * 70)
    
    all_data = []
    
    # Collect data for each season
    for year in range(2021, 2026):
        print(f"\n📥 Collecting {year} season data...")
        
        try:
            df_year = pitching_stats(year, qual=0)  # Get all pitchers
            
            # Filter to 50+ IP
            df_year = df_year[df_year['IP'] >= 50].copy()
            
            df_year['Season'] = year
            all_data.append(df_year)
            
            print(f"   ✓ {len(df_year)} pitchers with 50+ IP")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    # Combine all seasons
    print(f"\n🔗 Combining all seasons...")
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"   Total records: {len(df_combined)}")
    
    # Engineer features
    print(f"\n🔬 Engineering features...")
    df_combined = engineer_features(df_combined)
    
    # Create target
    df_combined = create_target(df_combined)
    
    # Remove pitchers without next season data (2025 pitchers have no target yet)
    df_with_target = df_combined[df_combined['next_season_strikeouts'].notna()].copy()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total records: {len(df_combined)}")
    print(f"   With targets (2021-2024): {len(df_with_target)}")
    print(f"   2025 pitchers (no target): {len(df_combined) - len(df_with_target)}")
    print(f"   Unique pitchers: {df_combined['full_name'].nunique()}")
    print(f"   Seasons: {df_combined['season'].min()}-{df_combined['season'].max()}")
    
    # Check for key pitchers
    print(f"\n🔍 Checking for key pitchers:")
    for pitcher in ['Tyler Glasnow', 'Nestor Cortes', 'Gerrit Cole']:
        pitcher_data = df_combined[df_combined['full_name'].str.contains(pitcher, case=False, na=False)]
        if len(pitcher_data) > 0:
            seasons = pitcher_data['season'].tolist()
            print(f"   ✓ {pitcher}: {seasons}")
        else:
            print(f"   ✗ {pitcher}: Not found")
    
    # Save both versions
    print(f"\n💾 Saving datasets...")
    
    # Full dataset (including 2025 without targets)
    output_full = 'data/pitcher_season_averages_improved_full.csv'
    df_combined.to_csv(output_full, index=False)
    print(f"   ✓ Full dataset: {output_full}")
    print(f"     ({len(df_combined)} records, {len(df_combined.columns)} columns)")
    
    # Training dataset (only with targets, 2021-2024)
    output_train = 'data/pitcher_season_averages_improved.csv'
    df_with_target.to_csv(output_train, index=False)
    print(f"   ✓ Training dataset: {output_train}")
    print(f"     ({len(df_with_target)} records, {len(df_with_target.columns)} columns)")
    
    # Show comparison
    print(f"\n📈 Before vs After:")
    try:
        df_old = pd.read_csv(output_train.replace('.csv', '_backup.csv'))
        print(f"   Old: {len(df_old)} records")
    except:
        print(f"   Old: Unknown (no backup)")
    print(f"   New: {len(df_with_target)} records")
    
    print(f"\n{'='*70}")
    print(f"✅ Dataset updated successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
