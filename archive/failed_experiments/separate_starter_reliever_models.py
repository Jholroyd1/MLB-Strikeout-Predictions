"""
Train separate models for starters and relievers
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SEPARATE STARTER/RELIEVER MODELS")
print("="*80)

# Load data
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

# Feature list (59 features)
feature_columns = [
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

# Split into starters and relievers
df_starters = df[df['is_starter'] == 1].copy()
df_relievers = df[df['is_reliever'] == 1].copy()

print(f"\nDataset split:")
print(f"  Total records: {len(df)}")
print(f"  Starters: {len(df_starters)} ({len(df_starters)/len(df)*100:.1f}%)")
print(f"  Relievers: {len(df_relievers)} ({len(df_relievers)/len(df)*100:.1f}%)")

# ============================================================================
# TEST 1: SINGLE COMBINED MODEL (BASELINE)
# ============================================================================
print(f"\n{'='*80}")
print("TEST 1: SINGLE COMBINED MODEL (Baseline)")
print(f"{'='*80}")

X_combined = df[feature_columns]
y_combined = df['next_season_strikeouts']

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_combined = Ridge(alpha=1.0)
ridge_combined.fit(X_train_scaled, y_train)
y_pred_combined = ridge_combined.predict(X_test_scaled)

mae_combined = mean_absolute_error(y_test, y_pred_combined)
r2_combined = r2_score(y_test, y_pred_combined)

print(f"\nCombined Model Performance:")
print(f"  MAE: {mae_combined:.2f} strikeouts")
print(f"  R²:  {r2_combined:.4f}")

# Calculate performance by role
test_indices = X_test.index
test_df = df.iloc[test_indices].copy()
test_df['predicted'] = y_pred_combined
test_df['actual'] = y_test.values
test_df['error'] = test_df['actual'] - test_df['predicted']
test_df['abs_error'] = np.abs(test_df['error'])

starters_mask = test_df['is_starter'] == 1
relievers_mask = test_df['is_starter'] == 0

mae_starters_combined = test_df[starters_mask]['abs_error'].mean()
mae_relievers_combined = test_df[relievers_mask]['abs_error'].mean()

print(f"\n  By Role:")
print(f"    Starters (n={starters_mask.sum()}):  MAE {mae_starters_combined:.2f}")
print(f"    Relievers (n={relievers_mask.sum()}): MAE {mae_relievers_combined:.2f}")

# ============================================================================
# TEST 2: SEPARATE MODELS FOR STARTERS AND RELIEVERS
# ============================================================================
print(f"\n{'='*80}")
print("TEST 2: SEPARATE STARTER/RELIEVER MODELS")
print(f"{'='*80}")

# Train starter model
print(f"\nTraining STARTER model...")
X_starters = df_starters[feature_columns]
y_starters = df_starters['next_season_strikeouts']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_starters, y_starters, test_size=0.2, random_state=42
)

scaler_s = StandardScaler()
X_train_s_scaled = scaler_s.fit_transform(X_train_s)
X_test_s_scaled = scaler_s.transform(X_test_s)

ridge_starters = Ridge(alpha=1.0)
ridge_starters.fit(X_train_s_scaled, y_train_s)
y_pred_starters = ridge_starters.predict(X_test_s_scaled)

mae_starters = mean_absolute_error(y_test_s, y_pred_starters)
r2_starters = r2_score(y_test_s, y_pred_starters)

print(f"  Records: {len(df_starters)}")
print(f"  Test size: {len(y_test_s)}")
print(f"  MAE: {mae_starters:.2f} strikeouts")
print(f"  R²:  {r2_starters:.4f}")

# Train reliever model
print(f"\nTraining RELIEVER model...")
X_relievers = df_relievers[feature_columns]
y_relievers = df_relievers['next_season_strikeouts']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_relievers, y_relievers, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

ridge_relievers = Ridge(alpha=1.0)
ridge_relievers.fit(X_train_r_scaled, y_train_r)
y_pred_relievers = ridge_relievers.predict(X_test_r_scaled)

mae_relievers = mean_absolute_error(y_test_r, y_pred_relievers)
r2_relievers = r2_score(y_test_r, y_pred_relievers)

print(f"  Records: {len(df_relievers)}")
print(f"  Test size: {len(y_test_r)}")
print(f"  MAE: {mae_relievers:.2f} strikeouts")
print(f"  R²:  {r2_relievers:.4f}")

# Calculate weighted average MAE
total_test = len(y_test_s) + len(y_test_r)
weighted_mae = (mae_starters * len(y_test_s) + mae_relievers * len(y_test_r)) / total_test

print(f"\nWeighted Average MAE: {weighted_mae:.2f} strikeouts")

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")

print(f"\nSingle Combined Model:")
print(f"  Overall MAE: {mae_combined:.2f}")
print(f"  Starters MAE: {mae_starters_combined:.2f}")
print(f"  Relievers MAE: {mae_relievers_combined:.2f}")

print(f"\nSeparate Models:")
print(f"  Overall MAE: {weighted_mae:.2f}")
print(f"  Starters MAE: {mae_starters:.2f}")
print(f"  Relievers MAE: {mae_relievers:.2f}")

improvement_overall = ((mae_combined - weighted_mae) / mae_combined) * 100
improvement_starters = ((mae_starters_combined - mae_starters) / mae_starters_combined) * 100
improvement_relievers = ((mae_relievers_combined - mae_relievers) / mae_relievers_combined) * 100

print(f"\nImprovement with Separate Models:")
print(f"  Overall: {improvement_overall:+.2f}% ({mae_combined - weighted_mae:+.2f} strikeouts)")
print(f"  Starters: {improvement_starters:+.2f}% ({mae_starters_combined - mae_starters:+.2f} strikeouts)")
print(f"  Relievers: {improvement_relievers:+.2f}% ({mae_relievers_combined - mae_relievers:+.2f} strikeouts)")

if improvement_overall > 0:
    print(f"\n✅ SEPARATE MODELS IMPROVED by {improvement_overall:.2f}%!")
else:
    print(f"\n❌ Separate models worse by {abs(improvement_overall):.2f}%")

# ============================================================================
# FEATURE IMPORTANCE DIFFERENCES
# ============================================================================
print(f"\n{'='*80}")
print("TOP 10 FEATURES BY MODEL")
print(f"{'='*80}")

# Starter features
starter_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': np.abs(ridge_starters.coef_)
}).sort_values('importance', ascending=False)

print(f"\nSTARTER Model Top 10:")
for i, row in starter_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

# Reliever features
reliever_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': np.abs(ridge_relievers.coef_)
}).sort_values('importance', ascending=False)

print(f"\nRELIEVER Model Top 10:")
for i, row in reliever_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print(f"\n{'='*80}")
print("SAVING MODELS")
print(f"{'='*80}")

import pickle

models = {
    'starter_model': ridge_starters,
    'reliever_model': ridge_relievers,
    'starter_scaler': scaler_s,
    'reliever_scaler': scaler_r,
    'feature_columns': feature_columns,
    'performance': {
        'combined_mae': mae_combined,
        'starter_mae': mae_starters,
        'reliever_mae': mae_relievers,
        'weighted_mae': weighted_mae,
        'improvement_pct': improvement_overall
    }
}

with open('data/separate_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print(f"\n✓ Saved models to: data/separate_models.pkl")
print(f"  - Starter Ridge model")
print(f"  - Reliever Ridge model")
print(f"  - Scalers and feature list")
print(f"  - Performance metrics")

print(f"\n{'='*80}")
print("✓ SEPARATE MODELS ANALYSIS COMPLETE!")
print(f"{'='*80}")
