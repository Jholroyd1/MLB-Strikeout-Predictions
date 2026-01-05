"""
Test tree-based models (XGBoost, LightGBM) vs Ridge Regression
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
print("TESTING TREE-BASED MODELS")
print("="*80)

# Load data
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

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

X = df[feature_columns]
y = df['next_season_strikeouts']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset: {df.shape}")
print(f"Features: {len(feature_columns)}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# MODEL 1: Ridge Regression (Baseline)
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 1: RIDGE REGRESSION (Baseline)")
print(f"{'='*80}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"MAE:  {mae_ridge:.2f} strikeouts")
print(f"R²:   {r2_ridge:.4f}")

# ============================================================================
# MODEL 2: Random Forest
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 2: RANDOM FOREST")
print(f"{'='*80}")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE:  {mae_rf:.2f} strikeouts")
print(f"R²:   {r2_rf:.4f}")
print(f"Improvement vs Ridge: {((mae_ridge - mae_rf) / mae_ridge * 100):+.2f}%")

# ============================================================================
# MODEL 3: Gradient Boosting
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 3: GRADIENT BOOSTING")
print(f"{'='*80}")

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"MAE:  {mae_gb:.2f} strikeouts")
print(f"R²:   {r2_gb:.4f}")
print(f"Improvement vs Ridge: {((mae_ridge - mae_gb) / mae_ridge * 100):+.2f}%")

# ============================================================================
# MODEL 4: XGBoost (if available)
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 4: XGBOOST")
print(f"{'='*80}")

try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print(f"MAE:  {mae_xgb:.2f} strikeouts")
    print(f"R²:   {r2_xgb:.4f}")
    print(f"Improvement vs Ridge: {((mae_ridge - mae_xgb) / mae_ridge * 100):+.2f}%")
    
    xgb_available = True
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    xgb_available = False
    mae_xgb = None
    r2_xgb = None

# ============================================================================
# MODEL 5: LightGBM (if available)
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 5: LIGHTGBM")
print(f"{'='*80}")

try:
    import lightgbm as lgb
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    
    print(f"MAE:  {mae_lgb:.2f} strikeouts")
    print(f"R²:   {r2_lgb:.4f}")
    print(f"Improvement vs Ridge: {((mae_ridge - mae_lgb) / mae_ridge * 100):+.2f}%")
    
    lgb_available = True
except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
    lgb_available = False
    mae_lgb = None
    r2_lgb = None

# ============================================================================
# MODEL 6: Ensemble (Average of all models)
# ============================================================================
print(f"\n{'='*80}")
print("MODEL 6: ENSEMBLE (Weighted Average)")
print(f"{'='*80}")

predictions = [y_pred_ridge, y_pred_rf, y_pred_gb]
weights = [0.3, 0.3, 0.4]  # Slight preference for Gradient Boosting

if xgb_available:
    predictions.append(y_pred_xgb)
    weights = [0.2, 0.2, 0.3, 0.3]

if lgb_available:
    predictions.append(y_pred_lgb)
    if xgb_available:
        weights = [0.15, 0.15, 0.25, 0.225, 0.225]
    else:
        weights = [0.2, 0.2, 0.3, 0.3]

# Normalize weights
weights = np.array(weights) / sum(weights)

y_pred_ensemble = np.average(predictions, axis=0, weights=weights)

mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print(f"MAE:  {mae_ensemble:.2f} strikeouts")
print(f"R²:   {r2_ensemble:.4f}")
print(f"Improvement vs Ridge: {((mae_ridge - mae_ensemble) / mae_ridge * 100):+.2f}%")
print(f"\nEnsemble weights: {weights}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")

results = [
    ('Ridge Regression', mae_ridge, r2_ridge),
    ('Random Forest', mae_rf, r2_rf),
    ('Gradient Boosting', mae_gb, r2_gb),
]

if xgb_available:
    results.append(('XGBoost', mae_xgb, r2_xgb))
if lgb_available:
    results.append(('LightGBM', mae_lgb, r2_lgb))

results.append(('Ensemble', mae_ensemble, r2_ensemble))

# Sort by MAE (best first)
results.sort(key=lambda x: x[1])

print(f"\n{'Model':<20} {'MAE':>10} {'R²':>10} {'vs Ridge':>12}")
print("-" * 56)

for model_name, mae, r2 in results:
    improvement = ((mae_ridge - mae) / mae_ridge * 100)
    status = '🏆' if mae == results[0][1] else '  '
    print(f"{status}{model_name:<18} {mae:10.2f} {r2:10.4f} {improvement:+11.2f}%")

print("-" * 56)

best_model, best_mae, best_r2 = results[0]
print(f"\n🏆 BEST MODEL: {best_model}")
print(f"   MAE: {best_mae:.2f} ({((mae_ridge - best_mae) / mae_ridge * 100):+.2f}% vs Ridge)")
print(f"   R²:  {best_r2:.4f}")

if best_mae < mae_ridge:
    print(f"\n✅ Tree-based models IMPROVED by {((mae_ridge - best_mae) / mae_ridge * 100):.2f}%!")
else:
    print(f"\n❌ Ridge still best (tree models {((best_mae - mae_ridge) / mae_ridge * 100):+.2f}% worse)")

# ============================================================================
# FEATURE IMPORTANCE (Best Tree Model)
# ============================================================================
if best_model != 'Ridge Regression' and best_model != 'Ensemble':
    print(f"\n{'='*80}")
    print(f"TOP 20 FEATURES - {best_model.upper()}")
    print(f"{'='*80}")
    
    if best_model == 'Random Forest':
        importances = rf.feature_importances_
    elif best_model == 'Gradient Boosting':
        importances = gb.feature_importances_
    elif best_model == 'XGBoost' and xgb_available:
        importances = xgb_model.feature_importances_
    elif best_model == 'LightGBM' and lgb_available:
        importances = lgb_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

print(f"\n{'='*80}")
print("✓ TREE MODEL TESTING COMPLETE!")
print(f"{'='*80}")
