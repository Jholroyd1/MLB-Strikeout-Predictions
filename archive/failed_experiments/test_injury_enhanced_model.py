"""
Test the injury-enhanced strikeout prediction model.

Compares:
1. Baseline model (59 features, no injury data)
2. Injury-enhanced model (66 features, includes injury history)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Original 59 features
ORIGINAL_FEATURES = [
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

# New injury features
INJURY_FEATURES = [
    'missed_previous_season',
    'consecutive_active_years',
    'years_since_injury',
    'career_availability_rate',
    'ip_volatility',
    'coming_back_from_injury',
    'injury_risk_score',
]

def test_model(X, y, feature_names, model_name):
    """Train and evaluate a model."""
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge model
    model = Ridge(alpha=0.5)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance (top 10)
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_features': len(feature_names),
        'top_features': importance.head(10),
        'model': model,
        'scaler': scaler
    }

def main():
    print("=" * 80)
    print("TESTING INJURY-ENHANCED STRIKEOUT PREDICTION MODEL")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading datasets...")
    df_original = pd.read_csv('data/pitcher_season_averages_improved.csv')
    df_injury = pd.read_csv('data/pitcher_season_averages_with_injury_features.csv')
    
    print(f"   Original: {len(df_original)} records, {len(df_original.columns)} columns")
    print(f"   Enhanced: {len(df_injury)} records, {len(df_injury.columns)} columns")
    
    # Prepare features
    X_original = df_original[ORIGINAL_FEATURES]
    X_enhanced = df_injury[ORIGINAL_FEATURES + INJURY_FEATURES]
    y = df_injury['next_season_strikeouts']
    
    print(f"\n🔬 Testing configurations:")
    print(f"   Baseline: {len(ORIGINAL_FEATURES)} features (no injury data)")
    print(f"   Enhanced: {len(ORIGINAL_FEATURES) + len(INJURY_FEATURES)} features (with injury data)")
    
    # Test baseline model
    print("\n" + "=" * 80)
    print("MODEL 1: BASELINE (No Injury Features)")
    print("=" * 80)
    baseline_results = test_model(X_original, y, ORIGINAL_FEATURES, "Baseline")
    
    print(f"\n📊 Performance:")
    print(f"   MAE:  {baseline_results['mae']:.2f} strikeouts")
    print(f"   RMSE: {baseline_results['rmse']:.2f}")
    print(f"   R²:   {baseline_results['r2']:.4f}")
    
    print(f"\n🏆 Top 10 Features:")
    print(baseline_results['top_features'][['feature', 'coefficient']].to_string(index=False))
    
    # Test injury-enhanced model
    print("\n" + "=" * 80)
    print("MODEL 2: INJURY-ENHANCED (With 7 Injury Features)")
    print("=" * 80)
    enhanced_results = test_model(
        X_enhanced, y, 
        ORIGINAL_FEATURES + INJURY_FEATURES, 
        "Injury-Enhanced"
    )
    
    print(f"\n📊 Performance:")
    print(f"   MAE:  {enhanced_results['mae']:.2f} strikeouts")
    print(f"   RMSE: {enhanced_results['rmse']:.2f}")
    print(f"   R²:   {enhanced_results['r2']:.4f}")
    
    print(f"\n🏆 Top 10 Features:")
    print(enhanced_results['top_features'][['feature', 'coefficient']].to_string(index=False))
    
    # Check if injury features made it to top features
    injury_in_top = enhanced_results['top_features']['feature'].isin(INJURY_FEATURES).sum()
    print(f"\n   💡 {injury_in_top} injury features in top 10")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    mae_change = enhanced_results['mae'] - baseline_results['mae']
    mae_pct = (mae_change / baseline_results['mae']) * 100
    r2_change = enhanced_results['r2'] - baseline_results['r2']
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'MAE':<20} {baseline_results['mae']:<15.2f} {enhanced_results['mae']:<15.2f} {mae_change:>+7.2f} ({mae_pct:+.2f}%)")
    print(f"{'RMSE':<20} {baseline_results['rmse']:<15.2f} {enhanced_results['rmse']:<15.2f} {enhanced_results['rmse'] - baseline_results['rmse']:>+7.2f}")
    print(f"{'R²':<20} {baseline_results['r2']:<15.4f} {enhanced_results['r2']:<15.4f} {r2_change:>+7.4f}")
    print(f"{'Features':<20} {baseline_results['n_features']:<15} {enhanced_results['n_features']:<15} +{enhanced_results['n_features'] - baseline_results['n_features']}")
    
    # Verdict
    print("\n" + "=" * 80)
    if mae_change < -0.5:  # Improved by more than 0.5 K
        print("✅ RESULT: Injury features IMPROVE the model")
        print(f"   Recommendation: Use injury-enhanced model (MAE improved by {abs(mae_change):.2f} K)")
    elif mae_change > 0.5:  # Got worse by more than 0.5 K
        print("❌ RESULT: Injury features HURT the model")
        print(f"   Recommendation: Keep baseline model (MAE got worse by {mae_change:.2f} K)")
    else:
        print("➖ RESULT: Injury features have MINIMAL impact")
        print(f"   Recommendation: Either model works (difference: {abs(mae_change):.2f} K)")
    
    # Test on specific example (Tyler Glasnow)
    print("\n" + "=" * 80)
    print("EXAMPLE: Tyler Glasnow 2024 → 2026 Prediction")
    print("=" * 80)
    
    glasnow_data = df_injury[
        (df_injury['full_name'] == 'Tyler Glasnow') & 
        (df_injury['season'] == 2024)
    ]
    
    if len(glasnow_data) > 0:
        # Baseline prediction
        glasnow_baseline = glasnow_data[ORIGINAL_FEATURES]
        glasnow_baseline_scaled = baseline_results['scaler'].transform(glasnow_baseline)
        baseline_pred = baseline_results['model'].predict(glasnow_baseline_scaled)[0]
        
        # Enhanced prediction
        glasnow_enhanced = glasnow_data[ORIGINAL_FEATURES + INJURY_FEATURES]
        glasnow_enhanced_scaled = enhanced_results['scaler'].transform(glasnow_enhanced)
        enhanced_pred = enhanced_results['model'].predict(glasnow_enhanced_scaled)[0]
        
        print(f"\nTyler Glasnow 2024 season stats:")
        print(f"   IP: {glasnow_data['total_innings_pitched'].values[0]:.1f}")
        print(f"   K: {glasnow_data['total_strikeouts'].values[0]:.0f}")
        print(f"   K/9: {glasnow_data['k_per_9'].values[0]:.2f}")
        
        print(f"\nInjury features:")
        print(f"   Missed 2025: {glasnow_data['missed_previous_season'].values[0]}")
        print(f"   Consecutive active years: {glasnow_data['consecutive_active_years'].values[0]}")
        print(f"   Years since injury: {glasnow_data['years_since_injury'].values[0]}")
        print(f"   Career availability: {glasnow_data['career_availability_rate'].values[0]:.2f}")
        print(f"   Injury risk score: {glasnow_data['injury_risk_score'].values[0]:.3f}")
        
        print(f"\n🎯 2026 Predictions:")
        print(f"   Baseline model (no injury data): {baseline_pred:.0f} K")
        print(f"   Enhanced model (with injury data): {enhanced_pred:.0f} K")
        print(f"   Difference: {enhanced_pred - baseline_pred:+.0f} K")
        
        if enhanced_pred < baseline_pred - 5:
            print(f"\n   💡 Injury model adjusts DOWN by {baseline_pred - enhanced_pred:.0f} K")
            print(f"      (accounts for injury risk and missed 2025 season)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
