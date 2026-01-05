"""
Analyze model errors to identify next improvements
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

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

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

# Calculate errors
errors = y_test - y_pred
abs_errors = np.abs(errors)

# Merge with original data for analysis
test_df = df.iloc[y_test.index].copy()
test_df['predicted'] = y_pred
test_df['actual'] = y_test.values
test_df['error'] = errors.values
test_df['abs_error'] = abs_errors.values

print("="*80)
print("MODEL IMPROVEMENT OPPORTUNITIES")
print("="*80)

print(f"\nCurrent MAE: {mean_absolute_error(y_test, y_pred):.2f} strikeouts")

print("\n" + "="*80)
print("1. ERROR ANALYSIS BY PITCHER TYPE")
print("="*80)

# By role
for role in [('Starters', 1), ('Relievers', 0)]:
    role_mask = test_df['is_starter'] == role[1]
    if role_mask.sum() > 0:
        role_mae = test_df[role_mask]['abs_error'].mean()
        role_pct = (role_mask.sum() / len(test_df)) * 100
        print(f"\n{role[0]} (n={role_mask.sum()}, {role_pct:.1f}%):")
        print(f"  MAE: {role_mae:.2f}")
        print(f"  Bias: {test_df[role_mask]['error'].mean():+.2f}")

# By strikeout level
print("\n" + "-"*80)
print("By Strikeout Tier:")
bins = [0, 75, 125, 175, 250, 500]
labels = ['Low (0-75)', 'Medium (75-125)', 'High (125-175)', 'Elite (175-250)', 'Ace (250+)']
test_df['k_tier'] = pd.cut(test_df['actual'], bins=bins, labels=labels)

for tier in labels:
    tier_mask = test_df['k_tier'] == tier
    if tier_mask.sum() > 0:
        tier_mae = test_df[tier_mask]['abs_error'].mean()
        tier_bias = test_df[tier_mask]['error'].mean()
        print(f"  {tier:20s} (n={tier_mask.sum():3d}): MAE={tier_mae:5.2f}, Bias={tier_bias:+6.2f}")

print("\n" + "="*80)
print("2. CORRELATION WITH ERRORS")
print("="*80)

# Check what correlates with large errors
correlations = []
check_features = [
    'k_per_9', 'total_innings_pitched', 'age', 'season_era', 
    'swstr_pct', 'is_ace', 'is_starter', 'power_index',
    'xFIP', 'SIERA', 'stuff_plus', 'command_plus'
]

for feat in check_features:
    if feat in test_df.columns:
        corr = test_df[feat].corr(test_df['abs_error'])
        correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nFeatures most correlated with prediction errors:")
for feat, corr in correlations[:10]:
    print(f"  {feat:25s}: {corr:+.3f}")

print("\n" + "="*80)
print("3. WORST PREDICTIONS")
print("="*80)

worst = test_df.nlargest(10, 'abs_error')[['full_name', 'season', 'actual', 'predicted', 'error', 'is_starter', 'k_per_9', 'total_innings_pitched', 'age']]
print("\nTop 10 worst predictions:")
print(worst.to_string(index=False))

print("\n" + "="*80)
print("4. IMPROVEMENT SUGGESTIONS")
print("="*80)

suggestions = []

# Check for systematic biases
starter_bias = test_df[test_df['is_starter'] == 1]['error'].mean()
reliever_bias = test_df[test_df['is_starter'] == 0]['error'].mean()

if abs(starter_bias) > 3:
    suggestions.append(f"A. Starter bias ({starter_bias:+.1f} K): Add more starter-specific features or separate models")
    
if abs(reliever_bias) > 3:
    suggestions.append(f"B. Reliever bias ({reliever_bias:+.1f} K): Add more reliever-specific features or separate models")

# Check for age effects
young_bias = test_df[test_df['age'] < 27]['error'].mean() if (test_df['age'] < 27).sum() > 0 else 0
old_bias = test_df[test_df['age'] > 32]['error'].mean() if (test_df['age'] > 32).sum() > 0 else 0

if abs(young_bias) > 5:
    suggestions.append(f"C. Young pitcher bias ({young_bias:+.1f} K): Add prospect/development indicators")
    
if abs(old_bias) > 5:
    suggestions.append(f"D. Veteran bias ({old_bias:+.1f} K): Add aging/decline indicators")

# Check injury/health
ace_bias = test_df[test_df['is_ace'] == 1]['error'].mean() if (test_df['is_ace'] == 1).sum() > 0 else 0
if abs(ace_bias) > 5:
    suggestions.append(f"E. Ace pitcher bias ({ace_bias:+.1f} K): Add workload fatigue or injury risk features")

# Feature engineering ideas
print("\n📊 SUGGESTED IMPROVEMENTS:\n")

if suggestions:
    for sugg in suggestions:
        print(f"  {sugg}")
else:
    print("  ✓ No major systematic biases detected!")

print("\n💡 ADDITIONAL IDEAS:\n")
print("  F. Year-to-year change features (ΔK/9, ΔERA, ΔIP)")
print("  G. Team/park factors (pitcher-friendly parks, team defense)")
print("  H. Injury history indicators (previous season missed time)")
print("  I. Contract year motivation (free agency upcoming)")
print("  J. Platoon splits (vs LHB/RHB performance)")
print("  K. Month-by-month trends (improving vs declining during season)")
print("  L. Pitch mix diversity (how many different pitch types)")
print("  M. Previous season trend (hot finish vs cold finish)")
print("  N. Team quality (contender vs rebuilding, run support)")
print("  O. Multi-year rolling averages (3-year K/9 average)")

print("\n" + "="*80)
