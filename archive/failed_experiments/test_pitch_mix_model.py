"""
Test if adding pitch mix features improves model performance
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
print("TESTING PITCH MIX FEATURES")
print("="*80)

# Load both datasets
df_baseline = pd.read_csv('data/pitcher_season_averages_improved.csv')
df_with_pitch = pd.read_csv('data/pitcher_season_averages_with_pitch_mix.csv')

print(f"\nBaseline dataset:")
print(f"  Records: {len(df_baseline)}")
print(f"  Features: {len(df_baseline.columns)}")

print(f"\nWith pitch mix dataset:")
print(f"  Records: {len(df_with_pitch)}")
print(f"  Features: {len(df_with_pitch.columns)}")
print(f"  New features: {len(df_with_pitch.columns) - len(df_baseline.columns)}")

# Original 59 features
baseline_features = [
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

# New pitch mix features
pitch_mix_features = [
    'pitch_diversity', 'pitch_mix_entropy', 'primary_pitch_pct',
    'fastball_heavy', 'breaking_ball_pct',
    'FB%', 'SL%', 'CT%', 'CB%', 'CH%', 'SF%'
]

# Combined features
all_features = baseline_features + pitch_mix_features

# Filter to available features
baseline_features_avail = [f for f in baseline_features if f in df_baseline.columns]
all_features_avail = [f for f in all_features if f in df_with_pitch.columns]

print(f"\n{'='*80}")
print("MODEL 1: BASELINE (59 features)")
print(f"{'='*80}")

X_base = df_baseline[baseline_features_avail]
y_base = df_baseline['next_season_strikeouts']

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

ridge_base = Ridge(alpha=0.5)
ridge_base.fit(X_train_base_scaled, y_train_base)
y_pred_base = ridge_base.predict(X_test_base_scaled)

mae_base = mean_absolute_error(y_test_base, y_pred_base)
r2_base = r2_score(y_test_base, y_pred_base)

print(f"\nBaseline (59 features):")
print(f"  Training records: {len(X_train_base)}")
print(f"  Test records: {len(X_test_base)}")
print(f"  MAE: {mae_base:.2f}")
print(f"  R²: {r2_base:.4f}")

print(f"\n{'='*80}")
print("MODEL 2: WITH PITCH MIX (59 + 11 = 70 features)")
print(f"{'='*80}")

X_pitch = df_with_pitch[all_features_avail]
y_pitch = df_with_pitch['next_season_strikeouts']

X_train_pitch, X_test_pitch, y_train_pitch, y_test_pitch = train_test_split(
    X_pitch, y_pitch, test_size=0.2, random_state=42
)

scaler_pitch = StandardScaler()
X_train_pitch_scaled = scaler_pitch.fit_transform(X_train_pitch)
X_test_pitch_scaled = scaler_pitch.transform(X_test_pitch)

ridge_pitch = Ridge(alpha=0.5)
ridge_pitch.fit(X_train_pitch_scaled, y_train_pitch)
y_pred_pitch = ridge_pitch.predict(X_test_pitch_scaled)

mae_pitch = mean_absolute_error(y_test_pitch, y_pred_pitch)
r2_pitch = r2_score(y_test_pitch, y_pred_pitch)

print(f"\nWith pitch mix ({len(all_features_avail)} features):")
print(f"  Training records: {len(X_train_pitch)}")
print(f"  Test records: {len(X_test_pitch)}")
print(f"  MAE: {mae_pitch:.2f}")
print(f"  R²: {r2_pitch:.4f}")

# Calculate improvement
mae_improvement = ((mae_base - mae_pitch) / mae_base) * 100
r2_improvement = ((r2_pitch - r2_base) / r2_base) * 100

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\nMAE:")
print(f"  Baseline: {mae_base:.2f}")
print(f"  With pitch mix: {mae_pitch:.2f}")
print(f"  Change: {mae_improvement:+.2f}% ({mae_base - mae_pitch:+.2f} strikeouts)")

print(f"\nR²:")
print(f"  Baseline: {r2_base:.4f}")
print(f"  With pitch mix: {r2_pitch:.4f}")
print(f"  Change: {r2_improvement:+.2f}%")

if mae_improvement > 1:
    print(f"\n✅ SUCCESS! Pitch mix features improved performance by {mae_improvement:.2f}%")
    print(f"   New best MAE: {mae_pitch:.2f} strikeouts")
    
    # Show most important pitch mix features
    print(f"\n📊 Pitch mix feature coefficients (Ridge):")
    coefs = pd.DataFrame({
        'feature': all_features_avail,
        'coefficient': ridge_pitch.coef_
    })
    pitch_coefs = coefs[coefs['feature'].isin(pitch_mix_features)].sort_values('coefficient', key=abs, ascending=False)
    
    for _, row in pitch_coefs.iterrows():
        impact = "⬆️" if row['coefficient'] > 0 else "⬇️"
        print(f"   {impact} {row['feature']:<25s}: {row['coefficient']:>8.4f}")
    
elif mae_improvement > 0:
    print(f"\n✓ MARGINAL IMPROVEMENT: {mae_improvement:+.2f}%")
    print(f"   Pitch mix helps slightly")
else:
    print(f"\n⚠️  NO IMPROVEMENT: {mae_improvement:+.2f}%")
    print(f"   Pitch mix features didn't help")

print(f"\n{'='*80}")
