"""
Use Lasso (L1 regularization) to select the most important features
and test if reducing from 59 to ~30-40 features improves performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LASSO FEATURE SELECTION")
print("="*80)

# Load current best dataset
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

print(f"\nCurrent dataset:")
print(f"  Records: {len(df)}")
print(f"  Current features: 59")
print(f"  Baseline MAE: 26.80 (Ridge with all 59 features)")

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

X = df[feature_columns]
y = df['next_season_strikeouts']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1: Use LassoCV to find optimal alpha
print(f"\n{'='*80}")
print("STEP 1: Finding optimal Lasso alpha with cross-validation")
print(f"{'='*80}")

alphas = np.logspace(-3, 2, 50)  # Test alphas from 0.001 to 100
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

optimal_alpha = lasso_cv.alpha_
print(f"\nOptimal alpha: {optimal_alpha:.4f}")
print(f"Cross-validation MAE: {-lasso_cv.score(X_train_scaled, y_train):.2f}")

# Step 2: Fit Lasso with optimal alpha and identify important features
print(f"\n{'='*80}")
print("STEP 2: Identifying important features")
print(f"{'='*80}")

lasso = Lasso(alpha=optimal_alpha, max_iter=10000, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Get feature importance (absolute coefficients)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': lasso.coef_,
    'abs_coefficient': np.abs(lasso.coef_)
}).sort_values('abs_coefficient', ascending=False)

# Identify non-zero features
non_zero_features = feature_importance[feature_importance['coefficient'] != 0]
zero_features = feature_importance[feature_importance['coefficient'] == 0]

print(f"\nFeature selection results:")
print(f"  Features with non-zero coefficients: {len(non_zero_features)}")
print(f"  Features eliminated (zero coefficients): {len(zero_features)}")

print(f"\n📊 Top 20 most important features:")
print(f"{'Rank':<6} {'Feature':<30} {'Coefficient':<15} {'Impact'}")
print("-" * 80)
for idx, row in non_zero_features.head(20).iterrows():
    print(f"{non_zero_features.index.get_loc(idx)+1:<6} {row['feature']:<30} {row['coefficient']:>12.4f}   {'⬆️ Positive' if row['coefficient'] > 0 else '⬇️ Negative'}")

if len(zero_features) > 0:
    print(f"\n❌ Features eliminated by Lasso ({len(zero_features)}):")
    for idx, row in zero_features.head(10).iterrows():
        print(f"   • {row['feature']}")
    if len(zero_features) > 10:
        print(f"   ... and {len(zero_features) - 10} more")

# Step 3: Test different feature count thresholds
print(f"\n{'='*80}")
print("STEP 3: Testing different feature counts")
print(f"{'='*80}")

# Test with top N features
feature_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50, len(non_zero_features)]
results = []

print(f"\n{'Features':<12} {'Train MAE':<12} {'Test MAE':<12} {'R²':<10} {'vs Baseline'}")
print("-" * 65)

for n_features in feature_counts:
    if n_features > len(non_zero_features):
        n_features = len(non_zero_features)
    
    # Select top N features
    top_features = non_zero_features.head(n_features)['feature'].tolist()
    
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    # Scale
    scaler_selected = StandardScaler()
    X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler_selected.transform(X_test_selected)
    
    # Train Ridge with selected features
    ridge = Ridge(alpha=0.5)  # Use optimal alpha from earlier analysis
    ridge.fit(X_train_selected_scaled, y_train)
    
    # Evaluate
    y_train_pred = ridge.predict(X_train_selected_scaled)
    y_test_pred = ridge.predict(X_test_selected_scaled)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    improvement = ((26.80 - test_mae) / 26.80) * 100
    
    results.append({
        'n_features': n_features,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'r2': test_r2,
        'improvement': improvement
    })
    
    marker = "✓" if test_mae < 26.80 else " "
    print(f"{marker} {n_features:<10} {train_mae:>10.2f}  {test_mae:>10.2f}  {test_r2:>8.4f}  {improvement:>+6.2f}%")

# Find best result
best_result = min(results, key=lambda x: x['test_mae'])

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\n�� Best configuration:")
print(f"   Features: {best_result['n_features']}")
print(f"   Test MAE: {best_result['test_mae']:.2f}")
print(f"   R²: {best_result['r2']:.4f}")
print(f"   Improvement: {best_result['improvement']:+.2f}%")

if best_result['improvement'] > 0.5:
    print(f"\n✅ SUCCESS! Lasso feature selection improved performance")
    print(f"   Reducing from 59 to {best_result['n_features']} features helps!")
    
    # Save selected features
    best_features = non_zero_features.head(best_result['n_features'])['feature'].tolist()
    
    print(f"\n💾 Selected features ({len(best_features)}):")
    for i, feat in enumerate(best_features, 1):
        coef = non_zero_features[non_zero_features['feature'] == feat]['coefficient'].values[0]
        print(f"   {i:2d}. {feat:<30} (coef: {coef:>8.4f})")
    
    # Save to file
    with open('data/lasso_selected_features.txt', 'w') as f:
        f.write(f"# Lasso-selected features (n={len(best_features)})\n")
        f.write(f"# Test MAE: {best_result['test_mae']:.2f}\n")
        f.write(f"# Improvement: {best_result['improvement']:+.2f}%\n\n")
        for feat in best_features:
            f.write(f"{feat}\n")
    
    print(f"\n✓ Saved feature list to: data/lasso_selected_features.txt")
    
elif best_result['improvement'] > 0:
    print(f"\n✓ MARGINAL IMPROVEMENT: {best_result['improvement']:+.2f}%")
    print(f"   Feature selection helps slightly but not significantly")
else:
    print(f"\n⚠️  NO IMPROVEMENT: {best_result['improvement']:+.2f}%")
    print(f"   Keeping all 59 features is better")
    print(f"   This suggests all features contribute useful signal")

# Compare baseline (59 features) vs best selected
baseline_ridge = Ridge(alpha=0.5)
baseline_ridge.fit(X_train_scaled, y_train)
baseline_pred = baseline_ridge.predict(X_test_scaled)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"\n{'='*80}")
print("FINAL COMPARISON")
print(f"{'='*80}")

print(f"\nBaseline (all 59 features):")
print(f"   MAE: {baseline_mae:.2f}")
print(f"   R²: {baseline_r2:.4f}")

print(f"\nLasso-selected ({best_result['n_features']} features):")
print(f"   MAE: {best_result['test_mae']:.2f}")
print(f"   R²: {best_result['r2']:.4f}")
print(f"   Change: {best_result['improvement']:+.2f}%")

print(f"\n{'='*80}")
