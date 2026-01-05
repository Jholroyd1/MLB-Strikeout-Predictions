"""
Add pitch mix data from pybaseball Statcast
Features: Fastball%, Breaking ball%, Offspeed%, pitch usage trends
"""
import pandas as pd
import numpy as np
from pybaseball import pitching_stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADDING PITCH MIX FEATURES FROM FANGRAPHS")
print("="*80)

# Load current dataset
df = pd.read_csv('data/pitcher_season_averages_improved.csv')
print(f"\nCurrent dataset:")
print(f"  Records: {len(df)}")
print(f"  Seasons: {sorted(df['season'].unique())}")
print(f"  Current features: {len(df.columns)}")

# Get unique pitcher IDs and seasons
print(f"\n{'='*80}")
print("FETCHING PITCH MIX DATA")
print(f"{'='*80}")

all_pitch_data = []
seasons = sorted(df['season'].unique())

print(f"\nCollecting pitch mix for {len(seasons)} seasons...")

for year in seasons:
    print(f"  {year}...", end=" ", flush=True)
    try:
        # Get pitching stats with pitch mix (qual=0 gets all pitchers)
        stats = pitching_stats(year, qual=0)
        
        if stats is not None and len(stats) > 0:
            stats['season'] = year
            all_pitch_data.append(stats)
            print(f"✓ ({len(stats)} pitchers)")
        else:
            print(f"✗ (no data)")
    except Exception as e:
        print(f"✗ Error: {e}")

if not all_pitch_data:
    print("\n✗ No pitch data collected. Exiting.")
    exit(1)

# Combine all seasons
pitch_df = pd.concat(all_pitch_data, ignore_index=True)
print(f"\n✓ Total pitch records: {len(pitch_df)}")

print(f"\n{'='*80}")
print("ENGINEERING PITCH MIX FEATURES")
print(f"{'='*80}")

# Rename Name column to full_name for matching
if 'Name' in pitch_df.columns:
    pitch_df.rename(columns={'Name': 'full_name'}, inplace=True)

# Show available pitch mix columns
pitch_pct_cols = [c for c in pitch_df.columns if '%' in c and any(x in c for x in ['FB', 'SL', 'CT', 'CB', 'CH', 'SF', 'KN'])]
print(f"\nAvailable pitch mix columns: {len(pitch_pct_cols)}")
print(f"  {pitch_pct_cols[:10]}")

# Select key pitch mix features
# FB% = fastball, SL% = slider, CT% = cutter, CB% = curveball, CH% = changeup, SF% = splitter, KN% = knuckler
selected_pitch_cols = ['FB%', 'SL%', 'CT%', 'CB%', 'CH%', 'SF%']

print(f"\nCreating pitch mix features...")

# Create aggregated fastball percentage (FB% includes all fastball types)
pitch_summary = pitch_df.copy()

# Calculate pitch mix diversity (how many different pitches used)
available_pitch_cols = [c for c in selected_pitch_cols if c in pitch_df.columns]
if available_pitch_cols:
    pitch_summary['pitch_diversity'] = (pitch_df[available_pitch_cols] > 0).sum(axis=1)
    print(f"  ✓ Created pitch_diversity (# of pitch types)")

# Calculate pitch mix balance (entropy - higher = more balanced arsenal)
if available_pitch_cols:
    def calculate_entropy(row):
        probs = row[available_pitch_cols].values
        probs = np.array([p for p in probs if not pd.isna(p) and p > 0], dtype=float)
        if len(probs) == 0 or probs.sum() == 0:
            return 0.0
        probs = probs / probs.sum()  # Normalize
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    pitch_summary['pitch_mix_entropy'] = pitch_df.apply(calculate_entropy, axis=1)
    print(f"  ✓ Created pitch_mix_entropy (pitch mix balance)")

# Get primary pitch percentage (most-used pitch)
if available_pitch_cols:
    pitch_summary['primary_pitch_pct'] = pitch_df[available_pitch_cols].max(axis=1)
    print(f"  ✓ Created primary_pitch_pct (% of primary pitch)")

# Calculate fastball dependency (high FB% = fastball reliant)
if 'FB%' in pitch_df.columns:
    pitch_summary['fastball_heavy'] = (pitch_df['FB%'] > 0.5).astype(int)
    print(f"  ✓ Created fastball_heavy (FB% > 50%)")

# Calculate breaking ball usage (SL% + CB%)
if 'SL%' in pitch_df.columns and 'CB%' in pitch_df.columns:
    pitch_summary['breaking_ball_pct'] = pitch_df['SL%'].fillna(0) + pitch_df['CB%'].fillna(0)
    print(f"  ✓ Created breaking_ball_pct (SL% + CB%)")

# Select columns to merge
merge_cols = ['full_name', 'season']

# Add pitch mix features we created
new_features = ['pitch_diversity', 'pitch_mix_entropy', 'primary_pitch_pct', 'fastball_heavy', 'breaking_ball_pct']
for feat in new_features:
    if feat in pitch_summary.columns:
        merge_cols.append(feat)

# Add individual pitch percentages
for pitch_type in selected_pitch_cols:
    if pitch_type in pitch_df.columns:
        merge_cols.append(pitch_type)
        print(f"  ✓ Including {pitch_type}")

# Keep only columns that exist
merge_cols = [c for c in merge_cols if c in pitch_summary.columns]
pitch_summary = pitch_summary[merge_cols]

print(f"\n{'='*80}")
print("MERGING WITH EXISTING DATA")
print(f"{'='*80}")

# Normalize names for matching
def normalize_name(name):
    if pd.isna(name):
        return name
    # Remove accents, lowercase, remove punctuation
    import unicodedata
    name = str(name)
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = name.lower().replace('.', '').replace("'", '').strip()
    return name

df['name_normalized'] = df['full_name'].apply(normalize_name)
pitch_summary['name_normalized'] = pitch_summary['full_name'].apply(normalize_name)

# Merge on normalized name and season
print(f"\nMerging on normalized name + season...")
df_merged = df.merge(
    pitch_summary,
    on=['name_normalized', 'season'],
    how='left',
    suffixes=('', '_pitch')
)

# Drop duplicate full_name column and normalized name
df_merged.drop(columns=['name_normalized', 'full_name_pitch'], inplace=True, errors='ignore')

# Check merge success
pitch_cols_added = [c for c in merge_cols if c not in ['full_name', 'season', 'name_normalized']]
matched = df_merged[pitch_cols_added].notna().any(axis=1).sum()
print(f"\n✓ Matched {matched}/{len(df_merged)} records ({matched/len(df_merged)*100:.1f}%)")

# Fill missing values with median (for pitchers without Statcast data)
print(f"\nFilling missing pitch mix values with median...")
for col in pitch_cols_added:
    missing_before = df_merged[col].isna().sum()
    if missing_before > 0:
        median_val = df_merged[col].median()
        df_merged[col].fillna(median_val, inplace=True)
        print(f"  {col}: filled {missing_before} missing ({median_val:.2f})")

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")

print(f"\nFinal dataset:")
print(f"  Records: {len(df_merged)}")
print(f"  Original features: {len(df.columns)}")
print(f"  New pitch mix features: {len(pitch_cols_added)}")
print(f"  Total features: {len(df_merged.columns)}")

print(f"\n📊 New features added:")
for col in pitch_cols_added:
    print(f"   • {col}")

# Save
output_path = 'data/pitcher_season_averages_with_pitch_mix.csv'
df_merged.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")

# Show sample stats
print(f"\n{'='*80}")
print("SAMPLE PITCH MIX STATS")
print(f"{'='*80}")

for col in pitch_cols_added[:5]:
    if col in df_merged.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df_merged[col].mean():.2f}")
        print(f"  Std: {df_merged[col].std():.2f}")
        print(f"  Range: [{df_merged[col].min():.2f}, {df_merged[col].max():.2f}]")

print(f"\n{'='*80}")
