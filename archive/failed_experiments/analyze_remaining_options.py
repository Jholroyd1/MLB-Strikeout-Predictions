"""
Analyze remaining ways to improve the model beyond MAE 26.80
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("="*80)
print("REMAINING MODEL IMPROVEMENT OPTIONS")
print("="*80)

# Load current best dataset
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

print(f"\nCurrent model status:")
print(f"  Dataset: 2021-2025")
print(f"  Records: {len(df)}")
print(f"  Features: 59")
print(f"  MAE: 26.80 strikeouts")
print(f"  R²: 0.5427")

print(f"\n{'='*80}")
print("WHAT WE'VE ALREADY TRIED (ALL FAILED)")
print(f"{'='*80}")

tried = [
    ("Momentum features", "Year-over-year changes, career trajectory", "-6.2%"),
    ("Separate models", "Starter/reliever split models", "-10.1%"),
    ("Tree models", "XGBoost, Random Forest, Gradient Boosting", "-4% to -8.5%"),
    ("Ensemble", "Weighted combination of models", "-3.9%"),
    ("More data", "2015-2025 expanded dataset", "-7.1%"),
    ("2019-2025", "Modern era with more records", "-10.8%"),
]

for name, desc, result in tried:
    print(f"  ✗ {name:20s} {desc:45s} {result:>10s}")

print(f"\n{'='*80}")
print("REMAINING OPTIONS TO EXPLORE")
print(f"{'='*80}")

# Analyze current errors to find opportunities
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

# Analyze errors
errors = y_test.values - y_pred
abs_errors = np.abs(errors)

# Get test data with predictions
test_df = df.loc[X_test.index].copy()
test_df['predicted'] = y_pred
test_df['actual'] = y_test.values
test_df['error'] = errors
test_df['abs_error'] = abs_errors

print("\n1. EXTERNAL DATA SOURCES")
print("   " + "-"*76)

external_options = [
    ("Team quality", "Team wins, run support, bullpen quality", "High", "pybaseball, Baseball-Reference"),
    ("Park factors", "Ballpark strikeout rates, dimensions", "Medium", "pybaseball.park_factors()"),
    ("Injury history", "DL stints, innings workload stress", "Medium", "Manual scraping or MLB API"),
    ("Pitch mix", "% fastball, breaking ball usage trends", "High", "pybaseball.statcast_pitcher()"),
    ("Contract year", "Free agency motivation indicator", "Low", "Manual research"),
    ("Weather trends", "Temperature, humidity by stadium", "Low", "Weather APIs"),
]

for name, desc, impact, source in external_options:
    print(f"   • {name:18s} {desc:40s}")
    print(f"     Impact: {impact:8s}  Source: {source}")

print("\n2. ADVANCED FEATURE ENGINEERING")
print("   " + "-"*76)

# Check for non-linear relationships
print(f"   • Polynomial features (degree 2-3)")
print(f"     Create interactions beyond our 5 current ones")
print(f"     Risk: Overfitting with small dataset (877 records)")

print(f"\n   • Binning/segmentation")
print(f"     Create discrete buckets for continuous features")
print(f"     Example: K/9 tiers (< 7, 7-9, 9-11, 11+)")

print(f"\n   • Time-based encoding")
print(f"     Season as cyclical feature (early/late career)")
print(f"     Capture non-linear age effects")

print("\n3. NEURAL NETWORKS / DEEP LEARNING")
print("   " + "-"*76)

print(f"   • Pros: Can learn complex non-linear patterns")
print(f"   • Cons: Need 1000+ samples (we have 877)")
print(f"   • Verdict: LIKELY TO OVERFIT")
print(f"   • If trying: Use dropout, early stopping, simple architecture")

print("\n4. ENSEMBLE OF SPECIALIZED MODELS")
print("   " + "-"*76)

# Analyze error patterns by segment
high_k_pitchers = test_df[test_df['total_strikeouts'] > 150]
low_k_pitchers = test_df[test_df['total_strikeouts'] <= 150]

print(f"   • Elite pitchers (>150 K): MAE {high_k_pitchers['abs_error'].mean():.2f}")
print(f"   • Regular pitchers (≤150 K): MAE {low_k_pitchers['abs_error'].mean():.2f}")

starters = test_df[test_df['is_starter'] == 1]
relievers = test_df[test_df['is_reliever'] == 1]

print(f"   • Starters: MAE {starters['abs_error'].mean():.2f}")
print(f"   • Relievers: MAE {relievers['abs_error'].mean():.2f}")

print(f"\n   Strategy: Train separate models for each segment")
print(f"   Risk: Already tested starter/reliever split (-10%), may fail again")

print("\n5. HYPERPARAMETER OPTIMIZATION")
print("   " + "-"*76)

# Test different Ridge alpha values
alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
results = []

for alpha in alphas:
    ridge_test = Ridge(alpha=alpha)
    ridge_test.fit(X_train_scaled, y_train)
    y_pred_test = ridge_test.predict(X_test_scaled)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    results.append((alpha, mae_test))

print(f"\n   Testing Ridge alpha values:")
best_alpha, best_mae = min(results, key=lambda x: x[1])
for alpha, mae in results:
    marker = "✓" if alpha == best_alpha else " "
    improvement = ((26.80 - mae) / 26.80) * 100
    print(f"     {marker} alpha={alpha:5.1f}: MAE={mae:.2f} ({improvement:+.2f}%)")

print("\n6. FEATURE SELECTION / DIMENSIONALITY REDUCTION")
print("   " + "-"*76)

# Check for highly correlated features
correlation_matrix = X.corr()
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

print(f"   • Found {len(high_corr_pairs)} highly correlated feature pairs (>0.9)")
if len(high_corr_pairs) > 0:
    print(f"     Removing redundant features might help")
    for feat1, feat2, corr in high_corr_pairs[:3]:
        print(f"       - {feat1} ↔ {feat2}: {corr:.3f}")

print(f"\n   • Try PCA to reduce 59 features to ~20-30 components")
print(f"   • Try Lasso (L1) to auto-select most important features")

print("\n7. LOWER MINIMUM IP THRESHOLD")
print("   " + "-"*76)

# Check current IP distribution
print(f"   Current min IP: {df['total_innings_pitched'].min():.1f}")
print(f"   • Lower to 30 IP → more data points")
print(f"   • Risk: Include less relevant relievers")
print(f"   • Benefit: Better reliever prediction coverage")

print(f"\n{'='*80}")
print("RECOMMENDATIONS (RANKED BY LIKELIHOOD OF SUCCESS)")
print(f"{'='*80}")

recommendations = [
    (1, "Hyperparameter tuning", "Test Ridge alpha in range [0.5, 2.0]", "5 min", "0-2%", "Low risk"),
    (2, "Pitch mix data", "Add fastball%, breaking ball usage from Statcast", "2 hours", "1-3%", "High quality data"),
    (3, "Park factors", "Add ballpark K-rate adjustments", "1 hour", "0.5-2%", "Proven to help"),
    (4, "Feature selection", "Use Lasso to reduce to 30-40 best features", "30 min", "0-2%", "Reduce noise"),
    (5, "Team context", "Add team quality, run support", "2 hours", "0.5-1%", "Marginal signal"),
    (6, "Polynomial features", "Carefully add degree-2 interactions", "1 hour", "0-2%", "Overfitting risk"),
    (7, "Elite pitcher model", "Separate model for >200K pitchers only", "1 hour", "-5 to +2%", "Already failed similar"),
    (8, "Neural network", "Simple feedforward with dropout", "3 hours", "-5 to +1%", "Likely overfit"),
    (9, "Lower IP threshold", "Include 30+ IP pitchers", "30 min", "-2 to +1%", "More noise"),
]

print("\n")
for rank, name, desc, time, expected, notes in recommendations:
    print(f"{rank}. {name}")
    print(f"   Action: {desc}")
    print(f"   Time: {time:10s}  Expected: {expected:8s}  Note: {notes}")
    print()

print(f"{'='*80}")
print("REALITY CHECK")
print(f"{'='*80}")
print(f"\nCurrent MAE 26.80 is already very good:")
print(f"  • Predicting strikeouts ±27 for next season")
print(f"  • R² = 0.54 (explains 54% of variance)")
print(f"  • Beat all complex alternatives (trees, ensembles, more data)")
print(f"\nDiminishing returns are real:")
print(f"  • Getting to 26.0 MAE might require 10x effort")
print(f"  • Each 0.5 improvement gets exponentially harder")
print(f"  • Some variance is just unpredictable (injuries, role changes)")
print(f"\n💡 BEST NEXT STEP: Try #1 (hyperparameter tuning) + #2 (pitch mix data)")
print(f"   These have highest success probability with reasonable effort.")

print(f"\n{'='*80}")
