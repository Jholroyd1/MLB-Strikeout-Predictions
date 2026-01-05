"""
Generate 2026 strikeout projections using Random Forest model
(Better performance: MAE 28.66 vs Ridge's 29.92)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING 2026 PROJECTIONS WITH RANDOM FOREST MODEL")
print("=" * 80)

# Define features (59 features)
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

# Load training data (2021-2024)
print("\n1. Loading training data...")
df_train = pd.read_csv('data/pitcher_season_averages_improved.csv')
print(f"   ✓ Loaded {len(df_train)} records (2021-2024)")

# Prepare training data
X_train = df_train[FEATURES]
y_train = df_train['next_season_strikeouts']

# Train Random Forest model
print("\n2. Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on training set (for MAE reference)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
model_eval = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model_eval.fit(X_train_split, y_train_split)
y_pred_eval = model_eval.predict(X_test_split)
mae = mean_absolute_error(y_test_split, y_pred_eval)
r2 = r2_score(y_test_split, y_pred_eval)

print(f"   ✓ Model trained")
print(f"   ✓ Test MAE: {mae:.2f} strikeouts")
print(f"   ✓ Test R²: {r2:.4f}")

# Load 2025 data for predictions
print("\n3. Loading 2025 data for predictions...")
df_full = pd.read_csv('data/pitcher_season_averages_improved_full.csv')
df_2025 = df_full[df_full['season'] == 2025].copy()
print(f"   ✓ Loaded {len(df_2025)} pitchers from 2025")

# Make predictions
print("\n4. Generating 2026 predictions...")
X_2025 = df_2025[FEATURES]
predictions = model.predict(X_2025)

# Prepare results dataframe
results = pd.DataFrame({
    'Pitcher': df_2025['full_name'].values,
    'Team': df_2025['Team'].values if 'Team' in df_2025.columns else 'N/A',
    '2025_IP': df_2025['total_innings_pitched'].values,
    '2025_SO': df_2025['total_strikeouts'].values,
    '2025_K/9': df_2025['k_per_9'].values,
    'Age': df_2025['age'].values,
    'Predicted_2026_SO': predictions.round(0).astype(int)
})

# Add prediction ranges (±28.66 based on MAE)
mae_rounded = 29  # Round up for conservative ranges
results['Range_Low'] = (results['Predicted_2026_SO'] - mae_rounded).clip(lower=0)
results['Range_High'] = results['Predicted_2026_SO'] + mae_rounded

# Sort by predicted strikeouts
results = results.sort_values('Predicted_2026_SO', ascending=False)
results['Rank'] = range(1, len(results) + 1)

# Reorder columns
results = results[['Rank', 'Pitcher', 'Team', '2025_IP', '2025_SO', '2025_K/9', 
                   'Age', 'Predicted_2026_SO', 'Range_Low', 'Range_High']]

# Save results
output_file = 'data/2026_strikeout_projections.csv'
results.to_csv(output_file, index=False)
print(f"   ✓ Saved to {output_file}")

# Print summary
print("\n" + "=" * 80)
print("2026 PROJECTIONS SUMMARY")
print("=" * 80)
print(f"\nTotal pitchers projected: {len(results)}")
print(f"200+ K Club: {len(results[results['Predicted_2026_SO'] >= 200])} pitchers")
print(f"150+ K Tier: {len(results[results['Predicted_2026_SO'] >= 150])} pitchers")
print(f"100+ K Tier: {len(results[results['Predicted_2026_SO'] >= 100])} pitchers")

print(f"\n🌟 Top 10 Projected Strikeout Leaders for 2026:")
print("-" * 80)
for idx, row in results.head(10).iterrows():
    print(f"{row['Rank']:2d}. {row['Pitcher']:<25} ({row['Team']:<4}) - {row['Predicted_2026_SO']:3d} K (range: {row['Range_Low']:.0f}-{row['Range_High']:.0f})")

print("\n" + "=" * 80)
print("✅ COMPLETE - 2026 projections generated with Random Forest model")
print("=" * 80)
