# Quick Start Guide

## 🚀 Using the Strikeout Prediction Model

### 1. Load the Final Dataset

```python
import pandas as pd

df = pd.read_csv('data/pitcher_season_averages_improved.csv')
print(f"Records: {len(df)}")  # 877 pitcher-seasons
print(f"Columns: {len(df.columns)}")  # 115 total (59 features + target + metadata)
```

### 2. Train the Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Define the 59 features
features = [
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

X = df[features]
y = df['next_season_strikeouts']

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

# Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} strikeouts")  # Expected: ~26.78
print(f"R²: {r2:.4f}")                # Expected: ~0.543
```

### 3. Make Predictions

```python
# Example: Predict for a pitcher
pitcher_data = df[df['full_name'] == 'Gerrit Cole'].iloc[-1:][features]

# Scale and predict
pitcher_scaled = scaler.transform(pitcher_data)
predicted_k = model.predict(pitcher_scaled)[0]

print(f"Predicted strikeouts next season: {predicted_k:.0f}")
```

### 4. Feature Importance

```python
import numpy as np

# Get feature importance from coefficients
importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 most important features:")
print(importance.head(10))
```

## 📊 Expected Results

- **MAE**: ~26.78 strikeouts (±27 K prediction error)
- **R²**: ~0.543 (explains 54.3% of variance)
- **Training time**: < 1 second
- **Prediction time**: < 0.01 seconds

## 🎯 Use Cases

1. **Fantasy Baseball**: Predict next season K totals for draft prep
2. **Team Analytics**: Project pitcher performance for contract decisions
3. **Scouting**: Identify breakout candidates based on peripherals
4. **Research**: Understand what drives strikeout performance

## ⚠️ Limitations

- **Only predicts next season** (not multiple years ahead)
- **Requires minimum 50 IP** in current season
- **MAE ±27 K** means predictions can be off by that much
- **No injury/role change** predictions included
- **Modern era only** (2021-2025 training data)

## �� Documentation

See [README.md](README.md) for full details on:
- Feature descriptions
- All experiments conducted
- Key learnings
- Future improvement ideas

## 💾 Saving/Loading the Model

```python
import pickle

# Save
with open('models/strikeout_predictor.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'features': features}, f)

# Load
with open('models/strikeout_predictor.pkl', 'rb') as f:
    saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']
    features = saved['features']
```

## 🔗 Related Files

- **Main notebook**: `Predict_Pitcher_Strikeouts.ipynb`
- **Training data**: `data/pitcher_season_averages_improved.csv`
- **Test scripts**: `tests/` directory
- **Experimental scripts**: `scripts/` directory

---

**Questions?** Check the [full README](README.md) or review the Jupyter notebook for detailed analysis.
