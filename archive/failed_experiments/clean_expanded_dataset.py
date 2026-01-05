"""
Clean the expanded dataset by filling missing values
"""
import pandas as pd
import numpy as np

print("Loading expanded dataset...")
df = pd.read_csv('data/pitcher_season_averages_improved_2015_2025.csv')

print(f"Records: {len(df)}")
print(f"\nMissing values before cleaning:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(15))

# Fill missing values with reasonable defaults
# Stuff+ and Command+ (100 is league average)
df['stuff_plus'].fillna(100, inplace=True)
df['command_plus'].fillna(100, inplace=True)

# Other metrics - fill with 0 or median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        # For percentage fields, use 0
        if '%' in col or 'pct' in col.lower():
            df[col].fillna(0, inplace=True)
        # For others, use median
        else:
            df[col].fillna(df[col].median(), inplace=True)

print(f"\nMissing values after cleaning:")
missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
if len(missing_after) > 0:
    print(missing_after)
else:
    print("✓ No missing values!")

# Save cleaned version
output_path = 'data/pitcher_season_averages_improved_2015_2025.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Saved cleaned dataset to: {output_path}")
print(f"  Total records: {len(df)}")
print(f"  Features: {len(df.columns)}")
