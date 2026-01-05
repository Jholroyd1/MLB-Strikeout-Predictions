"""
Analyze if adding more seasons would improve the model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANALYZING VALUE OF MORE DATA")
print("="*80)

# Load current dataset
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

print(f"\nCurrent dataset:")
print(f"  Total records: {len(df)}")
print(f"  Seasons covered: {df['season'].min()} - {df['season'].max()}")
print(f"  Unique pitchers: {df['player_id'].nunique()}")

# Breakdown by season
print(f"\n{'='*80}")
print("RECORDS BY SEASON")
print(f"{'='*80}")

season_counts = df['season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count:4d} records")

print(f"\nTotal: {len(df)} records across {len(season_counts)} seasons")

# Check database for additional available seasons
print(f"\n{'='*80}")
print("CHECKING DATABASE FOR ADDITIONAL DATA")
print(f"{'='*80}")

import sqlite3
conn = sqlite3.connect('mlb_data.db')

# Check what seasons are in the database
query = """
SELECT season, COUNT(*) as count
FROM pitcher_game_logs
GROUP BY season
ORDER BY season
"""
db_seasons = pd.read_sql(query, conn)

print(f"\nSeasons available in database:")
for _, row in db_seasons.iterrows():
    in_model = '✓' if row['season'] in df['season'].values else '✗'
    print(f"  {in_model} {row['season']}: {row['count']:,} game records")

available_seasons = set(db_seasons['season'].values)
used_seasons = set(df['season'].values)
unused_seasons = available_seasons - used_seasons

if unused_seasons:
    print(f"\n📊 Unused seasons: {sorted(unused_seasons)}")
    print(f"   Potential additional records: ~{len(df) * len(unused_seasons) / len(used_seasons):.0f}")
else:
    print(f"\n✓ All available seasons already used")

conn.close()

# Learning curve analysis (does more data help?)
print(f"\n{'='*80}")
print("LEARNING CURVE ANALYSIS")
print(f"{'='*80}")

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

# Test with different training sizes
train_sizes = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = []

print(f"\nTesting model performance with different training set sizes:")
print(f"{'Train %':<10} {'Train Size':<12} {'Test MAE':<12} {'Improvement':>12}")
print("-" * 50)

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    results.append((train_size, len(X_train), mae))
    
    improvement = ""
    if len(results) > 1:
        prev_mae = results[-2][2]
        pct_change = ((prev_mae - mae) / prev_mae) * 100
        improvement = f"{pct_change:+.2f}%"
    
    print(f"{train_size*100:>6.0f}%     {len(X_train):>6d}       {mae:>8.2f}    {improvement:>12s}")

print("-" * 50)

# Estimate benefit of more data
print(f"\n{'='*80}")
print("PROJECTIONS")
print(f"{'='*80}")

current_records = len(df)
current_mae = results[-1][2]

# Estimate MAE improvement rate
mae_improvements = []
for i in range(1, len(results)):
    prev_mae = results[i-1][2]
    curr_mae = results[i][2]
    improvement_per_100 = ((prev_mae - curr_mae) / prev_mae) * 100 / (results[i][1] - results[i-1][1]) * 100
    mae_improvements.append(improvement_per_100)

avg_improvement_per_100_records = np.mean(mae_improvements) if mae_improvements else 0

print(f"\nCurrent performance:")
print(f"  Records: {current_records}")
print(f"  MAE: {current_mae:.2f}")
print(f"  Average improvement per 100 records: {avg_improvement_per_100_records:.3f}%")

# Projections
projections = [
    (500, "Adding ~100 more records"),
    (1000, "Adding ~500 more records (3-4 seasons)"),
    (2000, "Adding ~1500 records (doubling dataset)"),
]

print(f"\n📈 Projected improvements:")
for total_records, desc in projections:
    if total_records <= current_records:
        continue
    additional = total_records - current_records
    estimated_improvement = avg_improvement_per_100_records * (additional / 100)
    estimated_mae = current_mae * (1 - estimated_improvement / 100)
    
    print(f"\n  {desc}:")
    print(f"    Total records: {total_records}")
    print(f"    Estimated MAE: {estimated_mae:.2f} ({estimated_improvement:+.2f}%)")
    print(f"    Absolute improvement: {current_mae - estimated_mae:.2f} strikeouts")

# Diminishing returns analysis
print(f"\n{'='*80}")
print("DIMINISHING RETURNS ANALYSIS")
print(f"{'='*80}")

mae_decreases = []
for i in range(1, len(results)):
    mae_decrease = results[i-1][2] - results[i][2]
    mae_decreases.append(mae_decrease)

if mae_decreases:
    avg_decrease = np.mean(mae_decreases)
    trend = "decreasing" if mae_decreases[-1] < mae_decreases[0] else "stable"
    
    print(f"\nMAE improvement per step: {mae_decreases}")
    print(f"Average: {avg_decrease:.3f} strikeouts per step")
    print(f"Trend: {trend}")
    
    if trend == "decreasing":
        print(f"\n⚠️  DIMINISHING RETURNS: Each additional 10% of data provides less benefit")
    else:
        print(f"\n✓ STABLE RETURNS: More data should continue to help")

# Final recommendation
print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")

if len(unused_seasons) > 0:
    potential_improvement = avg_improvement_per_100_records * (len(df) * len(unused_seasons) / len(used_seasons)) / 100
    print(f"\n✅ YES, EXPAND TO MORE SEASONS!")
    print(f"\n   Unused seasons: {sorted(unused_seasons)}")
    print(f"   Estimated improvement: {potential_improvement:.2f}% ({current_mae * potential_improvement/100:.2f} strikeouts)")
    print(f"   Projected MAE: {current_mae - current_mae * potential_improvement/100:.2f}")
    print(f"\n   📊 Worth it if:")
    print(f"      • You want better accuracy (even 0.5-1 strikeout improvement matters)")
    print(f"      • More training data = more robust predictions")
    print(f"      • Can capture longer-term trends and career arcs")
elif avg_improvement_per_100_records > 0.1:
    print(f"\n✅ YES, MORE DATA HELPS!")
    print(f"   Current: {current_records} records")
    print(f"   Each 100 records improves by ~{avg_improvement_per_100_records:.3f}%")
    print(f"   Recommendation: Expand to more seasons or lower minimum IP threshold")
else:
    print(f"\n⚠️  MARGINAL BENEFIT")
    print(f"   Current: {current_records} records")
    print(f"   Improvement per 100 records: ~{avg_improvement_per_100_records:.3f}%")
    print(f"   Diminishing returns setting in")
    print(f"   Recommendation: Focus on feature engineering instead")

print(f"\n{'='*80}")
