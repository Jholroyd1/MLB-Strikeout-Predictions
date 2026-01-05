"""
Test model performance with momentum features
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
print("TESTING MOMENTUM FEATURES")
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

# NEW momentum features (31 features)
momentum_features = [
    'delta_k9', 'delta_swstr', 'delta_ip', 'delta_era', 'delta_xfip', 'delta_siera',
    'delta_stuff', 'delta_command', 'pct_change_k',
    'is_improving', 'is_declining', 'is_breakout', 'is_collapse',
    'stuff_improving', 'command_improving', 'workload_increasing', 'workload_decreasing',
    'missed_time_last_year', 'coming_off_injury',
    'career_high_k', 'career_high_k9', 'has_hit_200k', 'has_hit_11k9',
    'years_experience', 'is_rookie_level', 'is_established',
    'k9_rolling_std', 'is_volatile', 'k9_3yr_avg', 'swstr_3yr_avg', 'momentum_score'
]

# Combined features
all_features = baseline_features + momentum_features

print(f"\nFeature counts:")
print(f"  Baseline: {len(baseline_features)} features")
print(f"  Momentum: {len(momentum_features)} NEW features")
print(f"  Total: {len(all_features)} features")

# Test 1: Baseline model (without momentum)
print(f"\n{'='*80}")
print("TEST 1: BASELINE MODEL (No Momentum Features)")
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

# Test 2: With momentum features
print(f"\n{'='*80}")
print("TEST 2: WITH MOMENTUM FEATURES")
print(f"{'='*80}")

X_momentum = df_momentum[all_features]
y_momentum = df_momentum['next_season_strikeouts']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_momentum, y_momentum, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

ridge_momentum = Ridge(alpha=1.0)
ridge_momentum.fit(X_train2_scaled, y_train2)
y_pred_momentum = ridge_momentum.predict(X_test2_scaled)

mae_momentum = mean_absolute_error(y_test2, y_pred_momentum)
r2_momentum = r2_score(y_test2, y_pred_momentum)

print(f"MAE:  {mae_momentum:.2f} strikeouts")
print(f"R²:   {r2_momentum:.4f}")

# Comparison
print(f"\n{'='*80}")
print("IMPROVEMENT SUMMARY")
print(f"{'='*80}")

improvement_mae = ((mae_base - mae_momentum) / mae_base) * 100
improvement_r2 = ((r2_momentum - r2_base) / r2_base) * 100

print(f"\nBaseline (59 features):")
print(f"  MAE: {mae_base:.2f}")
print(f"  R²:  {r2_base:.4f}")

print(f"\nWith Momentum (90 features):")
print(f"  MAE: {mae_momentum:.2f}")
print(f"  R²:  {r2_momentum:.4f}")

print(f"\nImprovement:")
print(f"  MAE: {improvement_mae:+.2f}% ({mae_base - mae_momentum:+.2f} strikeouts)")
print(f"  R²:  {improvement_r2:+.2f}%")

if improvement_mae > 0:
    print(f"\n✅ MOMENTUM FEATURES IMPROVED MODEL by {improvement_mae:.2f}%!")
else:
    print(f"\n❌ Model got worse by {abs(improvement_mae):.2f}%")

# Feature importance analysis
print(f"\n{'='*80}")
print("TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*80}")

feature_importance = pd.DataFrame({
    'feature': all_features,
    'coefficient': np.abs(ridge_momentum.coef_)
})
feature_importance = feature_importance.sort_values('coefficient', ascending=False)

print("\nTop 20 features:")
for i, row in feature_importance.head(20).iterrows():
    is_new = '🆕' if row['feature'] in momentum_features else '  '
    print(f"  {is_new} {i+1:2d}. {row['feature']:30s} {row['coefficient']:.4f}")

# Count momentum features in top 20
momentum_in_top20 = len([f for f in feature_importance.head(20)['feature'] if f in momentum_features])
print(f"\n📊 {momentum_in_top20} momentum features in top 20 ({momentum_in_top20/20*100:.0f}%)")

print(f"\n{'='*80}")
