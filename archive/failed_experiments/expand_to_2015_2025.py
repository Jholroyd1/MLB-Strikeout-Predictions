#!/usr/bin/env python3
"""
Expand dataset to 2015-2025 seasons
Steps:
1. Initialize database (if needed)
2. Collect data for 2015-2025 using pybaseball
3. Export pitcher season averages
4. Merge with Statcast data
5. Add improved features (ace indicators, role features, interactions)
"""

import subprocess
import sys
import os
import pandas as pd
from datetime import datetime
import time

def run_command(description, command):
    """Run a shell command and report results"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(command, cwd=os.path.dirname(os.path.dirname(__file__)))
    elapsed = time.time() - start_time
    
    status = "✓ SUCCESS" if result.returncode == 0 else "✗ FAILED"
    print(f"\n{status} - Completed in {elapsed:.1f} seconds")
    
    return result.returncode == 0

def collect_pybaseball_data():
    """Use pybaseball to collect 2015-2025 data directly"""
    print(f"\n{'='*80}")
    print("COLLECTING DATA WITH PYBASEBALL (2015-2025)")
    print(f"{'='*80}\n")
    
    script_content = '''
import pandas as pd
from pybaseball import pitching_stats
import warnings
warnings.filterwarnings('ignore')

print("Collecting pitching data for 2015-2025...")
print("This may take a few minutes...\\n")

all_data = []

for year in range(2015, 2026):  # 2015-2025
    print(f"Fetching {year}...", end=" ", flush=True)
    try:
        # Get pitching stats for the year (qual=0 gets all pitchers)
        data = pitching_stats(year, qual=0)
        
        if data is not None and len(data) > 0:
            data['Season'] = year
            all_data.append(data)
            print(f"✓ ({len(data)} pitchers)")
        else:
            print("✗ (no data)")
    except Exception as e:
        print(f"✗ Error: {e}")

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to minimum 20 IP
    combined_filtered = combined[combined['IP'] >= 20].copy()
    
    print(f"\\n{'='*80}")
    print(f"Total pitchers collected: {len(combined):,}")
    print(f"After filtering (IP >= 20): {len(combined_filtered):,}")
    print(f"{'='*80}\\n")
    
    # Save raw data
    combined_filtered.to_csv('data/pitcher_stats_2015_2025_raw.csv', index=False)
    print(f"✓ Saved to: data/pitcher_stats_2015_2025_raw.csv")
    
    # Show breakdown by season
    print("\\nBreakdown by season:")
    season_counts = combined_filtered['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count:4d} pitchers")
else:
    print("\\n✗ No data collected")
'''
    
    # Write and execute the script
    script_path = 'scripts/temp_collect_pybaseball.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    success = run_command(
        "Collecting pitcher stats with pybaseball",
        [sys.executable, script_path]
    )
    
    # Clean up temp script
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return success

def prepare_model_dataset():
    """Prepare the modeling dataset with all features"""
    print(f"\n{'='*80}")
    print("PREPARING MODEL-READY DATASET")
    print(f"{'='*80}\n")
    
    script_content = '''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading raw pitcher data...")
df = pd.read_csv('data/pitcher_stats_2015_2025_raw.csv')

print(f"Loaded {len(df)} pitcher-season records\\n")

# Rename columns to match our existing format
column_mapping = {
    'Season': 'season',
    'Name': 'full_name',
    'IP': 'total_innings_pitched',
    'SO': 'total_strikeouts',
    'G': 'games_pitched',
    'K/9': 'k_per_9',
    'BB/9': 'bb_per_9',
    'HR/9': 'hr_per_9',
    'H/9': 'h_per_9',
    'BB': 'total_walks',
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
}

# Rename columns that exist
for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)

# Calculate additional features
print("Engineering features...")

# K/BB ratio
df['k_bb_ratio'] = df['total_strikeouts'] / df['total_walks'].replace(0, 1)

# Strike percentage (estimate)
if 'strike_percentage' not in df.columns:
    df['strike_percentage'] = 0.65  # typical average

# Pitch efficiency
df['pitch_efficiency'] = df['total_innings_pitched'] / df['games_pitched']

# Age features
df['age_squared'] = df['age'] ** 2
df['is_prime_age'] = ((df['age'] >= 27) & (df['age'] <= 31)).astype(int)
df['is_young'] = (df['age'] < 27).astype(int)
df['is_veteran'] = (df['age'] > 31).astype(int)
df['age_from_peak'] = abs(df['age'] - 29)

# Derived metrics
df['k_minus_bb_pct'] = (df['total_strikeouts'] - df['total_walks']) / df['total_innings_pitched']
df['contact_quality'] = df.get('Hard%', 30) / 100
df['whiff_rate'] = df['swstr_pct'] / 100 if 'swstr_pct' in df.columns else 0.10
df['k_to_contact_ratio'] = df['total_strikeouts'] / (df['total_innings_pitched'] * 3)
df['k_upside'] = df['k_per_9'] * df['total_innings_pitched'] / 100
df['power_index'] = df['k_per_9'] * df['swstr_pct'] / 100 if 'swstr_pct' in df.columns else df['k_per_9']
df['consistency_score'] = 100 - df.get('BB/9', df['bb_per_9']) * 10

# Zone metrics
if 'Zone%' in df.columns and 'Contact%' in df.columns:
    df['zone_contact_diff'] = df['Zone%'] - df['Contact%']
else:
    df['zone_contact_diff'] = 0

df['true_outcomes_pct'] = (df['total_strikeouts'] + df['total_walks']) / (df['total_innings_pitched'] * 3)

# Ace indicators
df['is_ace'] = ((df['k_per_9'] >= 9.5) & (df['total_innings_pitched'] >= 150)).astype(int)
df['is_high_k_pitcher'] = (df['k_per_9'] >= 9.5).astype(int)
df['is_workhorse'] = (df['total_innings_pitched'] >= 150).astype(int)

# Log transform
df['log_total_strikeouts'] = np.log1p(df['total_strikeouts'])

# Role features (starter vs reliever)
df['is_starter'] = (df['total_innings_pitched'] / df['games_pitched'] >= 4).astype(int)
df['is_reliever'] = 1 - df['is_starter']

# Role interaction features
df['k9_x_starter'] = df['k_per_9'] * df['is_starter']
df['swstr_x_starter'] = df['swstr_pct'] * df['is_starter'] if 'swstr_pct' in df.columns else 0
df['ip_x_starter'] = df['total_innings_pitched'] * df['is_starter']

# General interactions
df['swstr_x_ip'] = df['swstr_pct'] * df['total_innings_pitched'] if 'swstr_pct' in df.columns else 0
df['k9_x_era'] = df['k_per_9'] * df['season_era']
df['age_x_ip'] = df['age'] * df['total_innings_pitched']
df['stuff_x_command'] = df.get('stuff_plus', 100) * df.get('command_plus', 100) / 100
df['workload_stress'] = df['total_innings_pitched'] / df['age']

# Create next season target
print("Creating next season targets...")
df = df.sort_values(['full_name', 'season'])

# For each pitcher, get their next season strikeouts
df['next_season_strikeouts'] = df.groupby('full_name')['total_strikeouts'].shift(-1)

# Remove rows without next season data
df_model = df.dropna(subset=['next_season_strikeouts']).copy()

print(f"\\nFinal dataset: {len(df_model)} records with next season targets")
print(f"Seasons: {sorted(df_model['season'].unique())}")
print(f"Features: {len([c for c in df_model.columns if c not in ['full_name', 'season', 'next_season_strikeouts']])}")

# Save
output_path = 'data/pitcher_season_averages_improved_2015_2025.csv'
df_model.to_csv(output_path, index=False)
print(f"\\n✓ Saved to: {output_path}")

# Show breakdown
print("\\nRecords by season:")
season_counts = df_model['season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count:4d}")
'''
    
    # Write and execute
    script_path = 'scripts/temp_prepare_dataset.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    success = run_command(
        "Preparing model-ready dataset",
        [sys.executable, script_path]
    )
    
    # Clean up
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return success

def main():
    """Main workflow"""
    print(f"\n{'='*80}")
    print("EXPANDING DATASET TO 2015-2025")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    # Step 1: Collect data with pybaseball
    print("\\nSTEP 1: Collecting pitcher data...")
    if not collect_pybaseball_data():
        print("\\n✗ Data collection failed. Exiting.")
        return False
    
    # Step 2: Prepare model dataset
    print("\\nSTEP 2: Preparing model dataset...")
    if not prepare_model_dataset():
        print("\\n✗ Dataset preparation failed. Exiting.")
        return False
    
    # Done!
    elapsed = time.time() - overall_start
    print(f"\\n{'='*80}")
    print(f"✓ EXPANSION COMPLETE!")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: data/pitcher_season_averages_improved_2015_2025.csv")
    print(f"{'='*80}\\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
