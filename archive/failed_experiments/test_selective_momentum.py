"""
Test model with ONLY the best momentum features (selective approach)
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
print("TESTING SELECTIVE MOMENTUM FEATURES")
print("="*80)

# Load datasets
df_baseline = pd.read_csv('data/pitcher_season_averages_improved.csv')
df_momentum = pd.read_csv('data/pitcher_season_averages_with_momentum.csv')

# Baseline features (59 features)
baseline_features = [
    'total_innings_pitched', 'total_strikeouts', 'games_pitched', 'total_pitches',
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

# SELECTIVE momentum features (only the best 10)
selective_momentum_features = [
    'k9_3yr_avg',           # Rolling average smooths noise
    'swstr_3yr_avg',        # Same for swstr
    'career_high_k9',       # Ceiling indicator
    'years_experience',     # Experience matters
    'delta_k9',             # Year-over-year momentum  
    'is_improving',         # Binary momentum
    'has_hit_200k',         # Elite proven track record
    'momentum_score',       # Composite momentum
    'is_breakout',          # Major improvement signal
    'coming_off_injury',    # Health flag
]

# Combined with selective features
selective_features = baseline_features + selective_momentum_features

print(f"\nFeature comparison:")
print(f"  Baseline: {len(baseline_features)} features")
print(f"  Adding: {len(selective_momentum_features)} SELECTIVE momentum features")
print(f"  Total: {len(selective_features)} features")

# Test 1: Baseline
print(f"\n{'='*80}")
print("TEST 1: BASELINE")
print(f"{'='*80}")

X_base = df_baseline[baseline_features]
y_base = df_baseline['next_season_strikeouts']

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_base = Ridge(alpha=1.0)
ridge_base.fit(X_train_scaled, y_train)
y_pred_base = ridge_base.predict(X_test_scaled)

mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print(f"MAE:  {mae_base:.2f} strikeouts")
print(f"R²:   {r2_base:.4f}")

# Test 2: With selective momentum
print(f"\n{'='*80}")
print("TEST 2: WITH SELECTIVE MOMENTUM FEATURES")
print(f"{'='*80}")

X_selective = df_momentum[selective_features]
y_selective = df_momentum['next_season_strikeouts']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_selective, y_selective, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

ridge_selective = Ridge(alpha=1.0)
ridge_selective.fit(X_train2_scaled, y_train2)
y_pred_selective = ridge_selective.predict(X_test2_scaled)

mae_selective = mean_absolute_error(y_test2, y_pred_selective)
r2_selective = r2_score(y_test2, y_pred_selective)

print(f"MAE:  {mae_selective:.2f} strikeouts")
print(f"R²:   {r2_selective:.4f}")

# Comparison
print(f"\n{'='*80}")
print("IMPROVEMENT SUMMARY")
print(f"{'='*80}")

improvement_mae = ((mae_base - mae_selective) / mae_base) * 100
improvement_r2 = ((r2_selective - r2_base) / r2_base) * 100

print(f"\nBaseline (59 features):")
print(f"  MAE: {mae_base:.2f}")
print(f"  R²:  {r2_base:.4f}")

print(f"\nWith Selective Momentum (69 features):")
print(f"  MAE: {mae_selective:.2f}")
print(f"  R²:  {r2_selective:.4f}")

print(f"\nImprovement:")
print(f"  MAE: {improvement_mae:+.2f}% ({mae_base - mae_selective:+.2f} strikeouts)")
print(f"  R²:  {improvement_r2:+.2f}%")

if improvement_mae > 0:
    print(f"\n✅ SELECTIVE MOMENTUM IMPROVED MODEL by {improvement_mae:.2f}%!")
else:
    print(f"\n❌ Model got worse by {abs(improvement_mae):.2f}%")
    
print(f"\n💡 Selective approach vs ALL momentum:")
print(f"   All 31 features: MAE 28.44 (-6.2% worse)")
print(f"   Best 10 features: MAE {mae_selective:.2f} ({improvement_mae:+.1f}% change)")

print(f"\n{'='*80}")
