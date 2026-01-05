"""
Generate projections for pitchers missing from 2025 season
Uses their most recent season (2024, 2023, or 2022) with time-away adjustments
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING PROJECTIONS FOR PITCHERS WITHOUT 2025 DATA")
print("=" * 80)

# Define features
FEATURES = [
    'total_innings_pitched', 'total_strikeouts', 'games_pitched',
    'k_per_9', 'bb_per_9', 'hr_per_9', 'h_per_9',
    'total_walks', 'k_bb_ratio', 'strike_percentage',
    'season_era', 'season_whip', 'fip', 'xFIP', 'SIERA',
    'swstr_pct', 'CSW%', 'Contact%', 'O-Swing%', 'Z-Swing%', 'Zone%', 'F-Strike%',
    'Hard%', 'Barrel%', 'batting_avg_against', 'lob_pct',
    'age', 'age_squared', 'is_prime_age', 'is_young', 'is_veteran', 'age_from_peak',
    'stuff_plus', 'command_plus', 'k_minus_bb_pct', 'contact_quality', 'whiff_rate',
    'zone_contact_diff', 'true_outcomes_pct', 'k_to_contact_ratio', 'k_upside',
    'pitch_efficiency', 'power_index', 'consistency_score',
    'is_ace', 'is_high_k_pitcher', 'is_workhorse',
    'log_total_strikeouts',
    'is_starter', 'is_reliever',
    'k9_x_starter', 'swstr_x_starter', 'ip_x_starter',
    'swstr_x_ip', 'k9_x_era', 'age_x_ip', 'stuff_x_command', 'workload_stress',
]

# Load full dataset
print("\n1. Loading data...")
df_full = pd.read_csv('data/pitcher_season_averages_improved_full.csv')

# Get pitchers by season
pitchers_2025 = set(df_full[df_full['season'] == 2025]['full_name'].unique())
pitchers_2024 = set(df_full[df_full['season'] == 2024]['full_name'].unique())
pitchers_2023 = set(df_full[df_full['season'] == 2023]['full_name'].unique())
pitchers_2022 = set(df_full[df_full['season'] == 2022]['full_name'].unique())

print(f"   ✓ 2025: {len(pitchers_2025)} pitchers")
print(f"   ✓ 2024: {len(pitchers_2024)} pitchers")
print(f"   ✓ 2023: {len(pitchers_2023)} pitchers")
print(f"   ✓ 2022: {len(pitchers_2022)} pitchers")

# Find missing pitchers
missing_from_2025 = (pitchers_2024 | pitchers_2023 | pitchers_2022) - pitchers_2025
print(f"\n   ℹ️  {len(missing_from_2025)} pitchers missing from 2025")

# Train model on 2021-2024 data
print("\n2. Training Random Forest model...")
df_train = df_full[df_full['season'] < 2025].copy()
df_train = df_train.dropna(subset=['next_season_strikeouts'])

X_train = df_train[FEATURES]
y_train = df_train['next_season_strikeouts']

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"   ✓ Model trained on {len(X_train)} records")

# Process missing pitchers
print("\n3. Processing pitchers without 2025 data...")
supplemental_projections = []

for pitcher_name in missing_from_2025:
    # Get pitcher's most recent season data
    pitcher_data = df_full[df_full['full_name'] == pitcher_name].copy()
    
    if len(pitcher_data) == 0:
        continue
    
    # Get most recent season (2024, 2023, or 2022)
    most_recent = pitcher_data.sort_values('season', ascending=False).iloc[0]
    last_season = int(most_recent['season'])
    years_away = 2025 - last_season
    
    # Only project if last played within 3 years
    if years_away > 3:
        continue
    
    # Skip if insufficient innings in last season
    if most_recent['total_innings_pitched'] < 30:
        continue
    
    try:
        # Prepare features for prediction
        X_predict = most_recent[FEATURES].values.reshape(1, -1)
        
        # Base prediction
        base_prediction = model.predict(X_predict)[0]
        
        # Apply time-away adjustment
        # Assume 5% decline per year away, plus age effect
        time_away_penalty = 0.05 * years_away
        age_adjustment = 0.02 if most_recent['age'] > 32 else 0  # Extra penalty for older pitchers
        
        adjustment_factor = 1 - time_away_penalty - age_adjustment
        adjusted_prediction = base_prediction * adjustment_factor
        
        # Calculate uncertainty (higher for more time away)
        base_mae = 29
        uncertainty = base_mae * (1 + 0.2 * years_away)  # +20% uncertainty per year
        
        supplemental_projections.append({
            'Pitcher': pitcher_name,
            'Team': most_recent['Team'] if 'Team' in most_recent else 'N/A',
            'Last_Season': last_season,
            'Years_Away': years_away,
            'Last_IP': most_recent['total_innings_pitched'],
            'Last_SO': most_recent['total_strikeouts'],
            'Last_K/9': most_recent['k_per_9'],
            'Age': int(most_recent['age']) + years_away,  # Current age
            'Base_Prediction': int(round(base_prediction)),
            'Adjusted_2026_SO': int(round(adjusted_prediction)),
            'Range_Low': max(0, int(round(adjusted_prediction - uncertainty))),
            'Range_High': int(round(adjusted_prediction + uncertainty)),
            'Confidence': 'Low' if years_away >= 2 else 'Medium',
            'Note': f'Based on {last_season} season, {years_away}yr away'
        })
    except Exception as e:
        print(f"   ⚠️  Could not project {pitcher_name}: {str(e)}")
        continue

# Create DataFrame
if len(supplemental_projections) > 0:
    df_supplemental = pd.DataFrame(supplemental_projections)
    df_supplemental = df_supplemental.sort_values('Adjusted_2026_SO', ascending=False)
    df_supplemental['Rank'] = range(1, len(df_supplemental) + 1)
    
    # Reorder columns
    df_supplemental = df_supplemental[['Rank', 'Pitcher', 'Team', 'Last_Season', 'Years_Away',
                                       'Last_IP', 'Last_SO', 'Last_K/9', 'Age',
                                       'Adjusted_2026_SO', 'Range_Low', 'Range_High',
                                       'Confidence', 'Note']]
    
    # Save
    output_file = 'data/2026_supplemental_projections.csv'
    df_supplemental.to_csv(output_file, index=False)
    
    print(f"   ✓ Generated projections for {len(df_supplemental)} pitchers")
    print(f"   ✓ Saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUPPLEMENTAL PROJECTIONS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal pitchers: {len(df_supplemental)}")
    print(f"1 year away (2024): {len(df_supplemental[df_supplemental['Years_Away'] == 1])}")
    print(f"2 years away (2023): {len(df_supplemental[df_supplemental['Years_Away'] == 2])}")
    print(f"3 years away (2022): {len(df_supplemental[df_supplemental['Years_Away'] == 3])}")
    
    print(f"\nConfidence levels:")
    print(f"Medium: {len(df_supplemental[df_supplemental['Confidence'] == 'Medium'])}")
    print(f"Low: {len(df_supplemental[df_supplemental['Confidence'] == 'Low'])}")
    
    print(f"\n🔍 Top 10 Supplemental Projections:")
    print("-" * 80)
    for idx, row in df_supplemental.head(10).iterrows():
        print(f"{row['Rank']:2d}. {row['Pitcher']:<25} - {row['Adjusted_2026_SO']:3d} K "
              f"(last: {row['Last_Season']}, {row['Years_Away']}yr away, {row['Confidence']} confidence)")
    
    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT NOTES")
    print("=" * 80)
    print("• These projections are LESS RELIABLE than main projections")
    print("• Based on older data with time-away adjustments")
    print("• 5% decline assumed per year away + age penalties")
    print("• Higher uncertainty ranges reflect increased risk")
    print("• Use with caution for pitchers 2+ years away")
    print("=" * 80)
    
else:
    print("\n   ℹ️  No qualifying pitchers found for supplemental projections")

print("\n✅ COMPLETE")
