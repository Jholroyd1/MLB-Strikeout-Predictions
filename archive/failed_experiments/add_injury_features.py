"""
Add injury and availability features to the pitcher dataset.

This script enhances the model by adding features that capture:
1. Previous season missed (binary indicator)
2. Consecutive seasons with 50+ IP
3. Years since last injury (missed season)
4. Career availability rate
5. Workload volatility (IP variance across seasons)
"""

import pandas as pd
import numpy as np
from pybaseball import pitching_stats

def calculate_injury_features(df):
    """
    Calculate injury-related features for each pitcher-season.
    
    Args:
        df: DataFrame with pitcher data including 'full_name', 'season', 'total_innings_pitched'
    
    Returns:
        DataFrame with added injury features
    """
    # Sort by pitcher and season
    df = df.sort_values(['full_name', 'season']).copy()
    
    # Initialize new features
    df['missed_previous_season'] = 0
    df['consecutive_active_years'] = 0
    df['years_since_injury'] = 99  # High number = never injured
    df['career_availability_rate'] = 1.0
    df['ip_volatility'] = 0.0
    df['coming_back_from_injury'] = 0
    
    # Process each pitcher
    for pitcher in df['full_name'].unique():
        pitcher_mask = df['full_name'] == pitcher
        pitcher_data = df[pitcher_mask].copy()
        
        seasons = pitcher_data['season'].values
        ips = pitcher_data['total_innings_pitched'].values
        
        for idx, (season, ip) in enumerate(zip(seasons, ips)):
            row_idx = pitcher_data.index[idx]
            
            # 1. Check if missed previous season
            if idx > 0:
                prev_season = seasons[idx - 1]
                # If there's a gap, they missed season(s)
                if season - prev_season > 1:
                    df.loc[row_idx, 'missed_previous_season'] = 1
                    df.loc[row_idx, 'coming_back_from_injury'] = 1
            
            # 2. Consecutive active years (50+ IP seasons in a row)
            consecutive = 1
            for j in range(idx - 1, -1, -1):
                if seasons[j] == seasons[j + 1] - 1:  # Consecutive season
                    consecutive += 1
                else:
                    break
            df.loc[row_idx, 'consecutive_active_years'] = consecutive
            
            # 3. Years since last injury (gap in seasons)
            years_since = 99
            for j in range(idx - 1, -1, -1):
                if seasons[j] != seasons[j + 1] - 1:
                    years_since = season - seasons[j] - 1
                    break
            df.loc[row_idx, 'years_since_injury'] = years_since
            
            # 4. Career availability rate
            # How many seasons they played out of possible seasons in their career
            if idx > 0:
                career_span = season - seasons[0]
                seasons_played = idx + 1
                availability = seasons_played / (career_span + 1) if career_span > 0 else 1.0
                df.loc[row_idx, 'career_availability_rate'] = availability
            
            # 5. IP volatility (standard deviation of IP across career)
            if idx >= 2:  # Need at least 3 seasons
                career_ips = ips[:idx + 1]
                volatility = np.std(career_ips)
                df.loc[row_idx, 'ip_volatility'] = volatility
    
    return df

def add_injury_risk_score(df):
    """
    Calculate a composite injury risk score based on multiple factors.
    
    Higher score = higher injury risk
    """
    df = df.copy()
    
    # Normalize factors (0-1 scale, higher = more risk)
    
    # Factor 1: Missed previous season (0 or 1)
    f1 = df['missed_previous_season']
    
    # Factor 2: Low consecutive active years (inverse, normalized)
    max_consecutive = df['consecutive_active_years'].max()
    f2 = 1 - (df['consecutive_active_years'] / max_consecutive)
    
    # Factor 3: Recent injury (inverse of years since injury, capped at 5)
    f3 = np.clip(5 - df['years_since_injury'], 0, 5) / 5
    
    # Factor 4: Low availability rate
    f4 = 1 - df['career_availability_rate']
    
    # Factor 5: High IP volatility (normalized)
    if df['ip_volatility'].max() > 0:
        f5 = df['ip_volatility'] / df['ip_volatility'].max()
    else:
        f5 = 0
    
    # Weighted combination (more weight to recent injuries)
    injury_risk_score = (
        0.35 * f1 +  # Missed last season is strongest indicator
        0.20 * f2 +  # Consecutive years active
        0.25 * f3 +  # Years since last injury
        0.15 * f4 +  # Career availability
        0.05 * f5    # IP volatility
    )
    
    df['injury_risk_score'] = injury_risk_score
    
    return df

def main():
    print("=" * 70)
    print("Adding Injury Features to Pitcher Dataset")
    print("=" * 70)
    
    # Load existing data
    print("\n📂 Loading current dataset...")
    df = pd.read_csv('data/pitcher_season_averages_improved.csv')
    print(f"   Loaded {len(df)} records")
    
    # Calculate injury features
    print("\n🔬 Calculating injury features...")
    df = calculate_injury_features(df)
    
    print("\n   New features added:")
    print("   • missed_previous_season: Binary indicator of gap year")
    print("   • consecutive_active_years: Seasons in a row with 50+ IP")
    print("   • years_since_injury: Years since last missed season")
    print("   • career_availability_rate: % of seasons played vs career span")
    print("   • ip_volatility: Standard deviation of IP across career")
    print("   • coming_back_from_injury: First season back after gap")
    
    # Add composite risk score
    print("\n📊 Calculating composite injury risk score...")
    df = add_injury_risk_score(df)
    print("   • injury_risk_score: 0-1 composite risk (0=low, 1=high)")
    
    # Show summary statistics
    print("\n📈 Feature Summary:")
    injury_features = [
        'missed_previous_season', 'consecutive_active_years', 
        'years_since_injury', 'career_availability_rate', 
        'ip_volatility', 'injury_risk_score'
    ]
    
    for feature in injury_features:
        mean_val = df[feature].mean()
        print(f"   {feature:<30} mean: {mean_val:.3f}")
    
    # Show examples of high-risk pitchers
    print("\n⚠️  Top 10 Highest Injury Risk Pitchers (2024-2025):")
    recent = df[df['season'].isin([2024, 2025])].copy()
    top_risk = recent.nlargest(10, 'injury_risk_score')[
        ['full_name', 'season', 'injury_risk_score', 'missed_previous_season', 
         'consecutive_active_years', 'career_availability_rate']
    ]
    print(top_risk.to_string(index=False))
    
    # Save enhanced dataset
    output_file = 'data/pitcher_season_averages_with_injury_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced dataset to: {output_file}")
    print(f"   Total columns: {len(df.columns)} (added 7 injury features)")
    print("=" * 70)

if __name__ == "__main__":
    main()
