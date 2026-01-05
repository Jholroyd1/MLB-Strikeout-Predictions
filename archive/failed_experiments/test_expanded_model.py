"""
Test Ridge model with expanded 2015-2025 dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING EXPANDED DATASET (2015-2025)")
print("="*80)

# Load both datasets for comparison
print("\n1. Loading datasets...")
df_old = pd.read_csv('data/pitcher_season_averages_improved.csv')
df_new = pd.read_csv('data/pitcher_season_averages_improved_2015_2025.csv')

print(f"\nOld dataset (2021-2025):")
print(f"  Records: {len(df_old)}")
print(f"  Seasons: {sorted(df_old['season'].unique())}")

print(f"\nNew dataset (2015-2025):")
print(f"  Records: {len(df_new)}")
print(f"  Seasons: {sorted(df_new['season'].unique())}")
print(f"  Increase: +{len(df_new) - len(df_old)} records (+{((len(df_new)/len(df_old))-1)*100:.1f}%)")

# Get feature columns (use the 59 features we know work)
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

# Check which features are available
available_features = [f for f in feature_columns if f in df_new.columns]
print(f"\n2. Features available: {len(available_features)}/{len(feature_columns)}")

# Test old dataset
print(f"\n{'='*80}")
print("TESTING OLD DATASET (BASELINE)")
print(f"{'='*80}")

X_old = df_old[available_features]
y_old = df_old['next_season_strikeouts']

X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42
)

scaler_old = StandardScaler()
X_train_old_scaled = scaler_old.fit_transform(X_train_old)
X_test_old_scaled = scaler_old.transform(X_test_old)

ridge_old = Ridge(alpha=1.0)
ridge_old.fit(X_train_old_scaled, y_train_old)
y_pred_old = ridge_old.predict(X_test_old_scaled)

mae_old = mean_absolute_error(y_test_old, y_pred_old)
r2_old = r2_score(y_test_old, y_pred_old)

print(f"\nOld Dataset Results:")
print(f"  Training records: {len(X_train_old)}")
print(f"  Test records: {len(X_test_old)}")
print(f"  MAE: {mae_old:.2f} strikeouts")
print(f"  R²: {r2_old:.4f}")

# Test new dataset
print(f"\n{'='*80}")
print("TESTING NEW EXPANDED DATASET")
print(f"{'='*80}")

X_new = df_new[available_features]
y_new = df_new['next_season_strikeouts']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

scaler_new = StandardScaler()
X_train_new_scaled = scaler_new.fit_transform(X_train_new)
X_test_new_scaled = scaler_new.transform(X_test_new)

ridge_new = Ridge(alpha=1.0)
ridge_new.fit(X_train_new_scaled, y_train_new)
y_pred_new = ridge_new.predict(X_test_new_scaled)

mae_new = mean_absolute_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)

print(f"\nNew Dataset Results:")
print(f"  Training records: {len(X_train_new)}")
print(f"  Test records: {len(X_test_new)}")
print(f"  MAE: {mae_new:.2f} strikeouts")
print(f"  R²: {r2_new:.4f}")

# Comparison
print(f"\n{'='*80}")
print("IMPROVEMENT ANALYSIS")
print(f"{'='*80}")

mae_improvement = ((mae_old - mae_new) / mae_old) * 100
r2_improvement = ((r2_new - r2_old) / r2_old) * 100

print(f"\nMAE Improvement: {mae_improvement:+.2f}%")
print(f"  Old: {mae_old:.2f} → New: {mae_new:.2f}")
print(f"  Absolute: {mae_old - mae_new:+.2f} strikeouts")

print(f"\nR² Improvement: {r2_improvement:+.2f}%")
print(f"  Old: {r2_old:.4f} → New: {r2_new:.4f}")

if mae_improvement > 1:
    print(f"\n✅ SIGNIFICANT IMPROVEMENT! Expanding to 2015-2025 helps.")
elif mae_improvement > 0:
    print(f"\n✓ MARGINAL IMPROVEMENT. Expanding helps slightly.")
else:
    print(f"\n⚠️  NO IMPROVEMENT. Old dataset may have had better quality.")

print(f"\n{'='*80}")
