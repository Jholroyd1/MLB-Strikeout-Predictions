"""
Test 2019-2025 dataset (modern era with full Statcast)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("="*80)
print("TESTING 2019-2025 DATASET (Modern Era)")
print("="*80)

# Load expanded dataset and filter to 2019+
df_full = pd.read_csv('data/pitcher_season_averages_improved_2015_2025.csv')
df_modern = df_full[df_full['season'] >= 2019].copy()

print(f"\n2019-2025 dataset:")
print(f"  Records: {len(df_modern)}")
print(f"  Seasons: {sorted(df_modern['season'].unique())}")

season_counts = df_modern['season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"    {season}: {count}")

# Features
feature_columns = [
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

X = df_modern[feature_columns]
y = df_modern['next_season_strikeouts']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'='*80}")
print("RESULTS (2019-2025)")
print(f"{'='*80}")
print(f"  Training records: {len(X_train)}")
print(f"  Test records: {len(X_test)}")
print(f"  MAE: {mae:.2f} strikeouts")
print(f"  R²: {r2:.4f}")

# Compare to original 2021-2025
baseline_mae = 26.80
baseline_r2 = 0.5427

improvement = ((baseline_mae - mae) / baseline_mae) * 100

print(f"\n{'='*80}")
print("COMPARISON TO 2021-2025 BASELINE")
print(f"{'='*80}")
print(f"\nMAE:")
print(f"  2021-2025 (baseline): {baseline_mae:.2f}")
print(f"  2019-2025 (expanded): {mae:.2f}")
print(f"  Change: {improvement:+.2f}%")

print(f"\nR²:")
print(f"  2021-2025 (baseline): {baseline_r2:.4f}")
print(f"  2019-2025 (expanded): {r2:.4f}")

records_added = len(df_modern) - 877
print(f"\nRecords added: +{records_added} ({len(df_modern)} total vs 877)")

if improvement > 1:
    print(f"\n✅ WORTH IT! 2019-2025 gives {improvement:.2f}% better MAE")
elif improvement > 0:
    print(f"\n✓ MARGINAL. 2019-2025 slightly better (+{improvement:.2f}%)")
else:
    print(f"\n✗ NOT WORTH IT. 2021-2025 is better ({improvement:+.2f}%)")

print(f"\n{'='*80}")
