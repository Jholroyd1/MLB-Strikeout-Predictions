"""
Add year-over-year momentum features to capture pitcher trends
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADDING YEAR-OVER-YEAR MOMENTUM FEATURES")
print("=" * 80)

# Load the improved dataset
print("\n1. Loading data...")
df = pd.read_csv('data/pitcher_season_averages_improved.csv')
print(f"   Records: {len(df)}")

# Sort by player and season for lag calculations
df_sorted = df.sort_values(['player_id', 'season']).reset_index(drop=True)

print("\n2. Creating momentum features...")

# Calculate previous season values for key metrics
lag_features = {
    'k_per_9': 'prev_k9',
    'swstr_pct': 'prev_swstr',
    'total_innings_pitched': 'prev_ip',
    'season_era': 'prev_era',
    'xFIP': 'prev_xfip',
    'SIERA': 'prev_siera',
    'total_strikeouts': 'prev_total_k',
    'stuff_plus': 'prev_stuff',
    'command_plus': 'prev_command'
}

for orig_col, prev_col in lag_features.items():
    df_sorted[prev_col] = df_sorted.groupby('player_id')[orig_col].shift(1)

# Calculate year-over-year changes (DELTA features)
print("   • Delta (Δ) features...")
df_sorted['delta_k9'] = df_sorted['k_per_9'] - df_sorted['prev_k9']
df_sorted['delta_swstr'] = df_sorted['swstr_pct'] - df_sorted['prev_swstr']
df_sorted['delta_ip'] = df_sorted['total_innings_pitched'] - df_sorted['prev_ip']
df_sorted['delta_era'] = df_sorted['season_era'] - df_sorted['prev_era']
df_sorted['delta_xfip'] = df_sorted['xFIP'] - df_sorted['prev_xfip']
df_sorted['delta_siera'] = df_sorted['SIERA'] - df_sorted['prev_siera']
df_sorted['delta_stuff'] = df_sorted['stuff_plus'] - df_sorted['prev_stuff']
df_sorted['delta_command'] = df_sorted['command_plus'] - df_sorted['prev_command']

# Percentage change for strikeouts
df_sorted['pct_change_k'] = ((df_sorted['total_strikeouts'] - df_sorted['prev_total_k']) / 
                              df_sorted['prev_total_k'].replace(0, np.nan)) * 100

# Binary indicators for significant changes
print("   • Momentum indicators...")
df_sorted['is_improving'] = (df_sorted['delta_k9'] > 0.5).astype(int)
df_sorted['is_declining'] = (df_sorted['delta_k9'] < -0.5).astype(int)
df_sorted['is_breakout'] = (df_sorted['delta_k9'] > 1.5).astype(int)  # Major improvement
df_sorted['is_collapse'] = (df_sorted['delta_k9'] < -1.5).astype(int)  # Major decline

# Stuff improvement indicator (rising talent)
df_sorted['stuff_improving'] = (df_sorted['delta_stuff'] > 5).astype(int)
df_sorted['command_improving'] = (df_sorted['delta_command'] > 5).astype(int)

# Workload change indicators
df_sorted['workload_increasing'] = (df_sorted['delta_ip'] > 30).astype(int)  # 30+ more IP
df_sorted['workload_decreasing'] = (df_sorted['delta_ip'] < -30).astype(int)  # 30+ fewer IP

# Injury/Health indicators (previous season)
print("   • Health/Injury indicators...")
df_sorted['missed_time_last_year'] = (df_sorted['prev_ip'] < 100).astype(int)
df_sorted['coming_off_injury'] = ((df_sorted['prev_ip'] > 0) & 
                                   (df_sorted['prev_ip'] < 80)).astype(int)

# Career trajectory features
print("   • Career trajectory features...")
# Has pitcher ever exceeded certain thresholds?
df_sorted['career_high_k'] = df_sorted.groupby('player_id')['total_strikeouts'].cummax()
df_sorted['career_high_k9'] = df_sorted.groupby('player_id')['k_per_9'].cummax()
df_sorted['has_hit_200k'] = (df_sorted['career_high_k'] >= 200).astype(int)
df_sorted['has_hit_11k9'] = (df_sorted['career_high_k9'] >= 11.0).astype(int)

# Years in MLB (experience)
df_sorted['years_experience'] = df_sorted.groupby('player_id').cumcount() + 1
df_sorted['is_rookie_level'] = (df_sorted['years_experience'] <= 2).astype(int)
df_sorted['is_established'] = (df_sorted['years_experience'] >= 5).astype(int)

# Consistency score (how stable is their K/9?)
print("   • Consistency metrics...")
df_sorted['k9_rolling_std'] = df_sorted.groupby('player_id')['k_per_9'].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
)
df_sorted['is_volatile'] = (df_sorted['k9_rolling_std'] > 1.5).astype(int)

# Multi-year rolling averages
print("   • Rolling averages...")
df_sorted['k9_3yr_avg'] = df_sorted.groupby('player_id')['k_per_9'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
df_sorted['swstr_3yr_avg'] = df_sorted.groupby('player_id')['swstr_pct'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Momentum strength (combine multiple signals)
print("   • Composite momentum score...")
df_sorted['momentum_score'] = (
    (df_sorted['delta_k9'] / df_sorted['delta_k9'].std()) * 0.4 +
    (df_sorted['delta_swstr'] / df_sorted['delta_swstr'].std()) * 0.3 +
    (df_sorted['delta_stuff'] / df_sorted['delta_stuff'].std()) * 0.2 +
    (-df_sorted['delta_era'] / df_sorted['delta_era'].std()) * 0.1  # Lower ERA is better
)

# Fill NaN values for first-year pitchers (no previous season)
print("\n3. Handling missing values for first-year pitchers...")
momentum_cols = [col for col in df_sorted.columns if 'delta_' in col or 'prev_' in col or 
                 col in ['pct_change_k', 'momentum_score', 'k9_rolling_std', 'k9_3yr_avg', 'swstr_3yr_avg']]

# For first-year pitchers, fill with 0 (no change) or current values
for col in momentum_cols:
    if 'prev_' in col:
        # Fill previous values with current values for first-year pitchers
        current_col = col.replace('prev_', '').replace('_k9', '_per_9').replace('_k', '')
        if current_col == 'total_k':
            current_col = 'total_strikeouts'
        elif current_col == 'stuff':
            current_col = 'stuff_plus'
        elif current_col == 'command':
            current_col = 'command_plus'
        
        if current_col in df_sorted.columns:
            df_sorted[col] = df_sorted[col].fillna(df_sorted[current_col])
    elif 'delta_' in col or col in ['pct_change_k', 'momentum_score']:
        # Fill deltas with 0 (no change for first year)
        df_sorted[col] = df_sorted[col].fillna(0)
    elif col in ['k9_rolling_std', 'k9_3yr_avg', 'swstr_3yr_avg']:
        # Fill rolling stats with current values
        if 'k9' in col:
            df_sorted[col] = df_sorted[col].fillna(df_sorted['k_per_9'])
        elif 'swstr' in col:
            df_sorted[col] = df_sorted[col].fillna(df_sorted['swstr_pct'])

print(f"   Filled {df_sorted[momentum_cols].isna().sum().sum()} missing values")

# Count new features
new_features = [
    'delta_k9', 'delta_swstr', 'delta_ip', 'delta_era', 'delta_xfip', 'delta_siera',
    'delta_stuff', 'delta_command', 'pct_change_k',
    'is_improving', 'is_declining', 'is_breakout', 'is_collapse',
    'stuff_improving', 'command_improving', 'workload_increasing', 'workload_decreasing',
    'missed_time_last_year', 'coming_off_injury',
    'career_high_k', 'career_high_k9', 'has_hit_200k', 'has_hit_11k9',
    'years_experience', 'is_rookie_level', 'is_established',
    'k9_rolling_std', 'is_volatile', 'k9_3yr_avg', 'swstr_3yr_avg', 'momentum_score'
]

print(f"\n4. Created {len(new_features)} new momentum features:")
print(f"   • Year-over-year changes: {len([f for f in new_features if 'delta_' in f])}")
print(f"   • Momentum indicators: {len([f for f in new_features if 'is_' in f])}")
print(f"   • Career trajectory: {len([f for f in new_features if 'career_' in f or 'has_' in f])}")
print(f"   • Rolling averages: {len([f for f in new_features if 'avg' in f or 'rolling' in f])}")
print(f"   • Health/workload: {len([f for f in new_features if 'missed_' in f or 'workload_' in f or 'injury' in f])}")

# Summary statistics
print(f"\n5. Momentum feature statistics:")
print(f"   • Improving pitchers: {df_sorted['is_improving'].sum()} ({df_sorted['is_improving'].mean()*100:.1f}%)")
print(f"   • Declining pitchers: {df_sorted['is_declining'].sum()} ({df_sorted['is_declining'].mean()*100:.1f}%)")
print(f"   • Breakout pitchers: {df_sorted['is_breakout'].sum()} ({df_sorted['is_breakout'].mean()*100:.1f}%)")
print(f"   • Collapse pitchers: {df_sorted['is_collapse'].sum()} ({df_sorted['is_collapse'].mean()*100:.1f}%)")
print(f"   • Has hit 200K: {df_sorted['has_hit_200k'].sum()} ({df_sorted['has_hit_200k'].mean()*100:.1f}%)")
print(f"   • Coming off injury: {df_sorted['coming_off_injury'].sum()} ({df_sorted['coming_off_injury'].mean()*100:.1f}%)")

# Save enhanced dataset
print(f"\n6. Saving enhanced dataset...")
output_path = 'data/pitcher_season_averages_with_momentum.csv'
df_sorted.to_csv(output_path, index=False)

print(f"   Saved to: {output_path}")
print(f"   New shape: {df_sorted.shape}")
print(f"   Total columns: {len(df_sorted.columns)} (was {len(df.columns)})")
print(f"   Added {len(df_sorted.columns) - len(df.columns)} columns")

print("\n" + "=" * 80)
print("✓ MOMENTUM FEATURES CREATED SUCCESSFULLY!")
print("=" * 80)
