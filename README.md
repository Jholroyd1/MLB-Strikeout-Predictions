# MLB Pitcher Strikeout Prediction

Machine learning model to predict next season strikeouts for MLB pitchers.

## 📊 Model Performance

**Final Model**: Ridge Regression (alpha=0.5)
- **MAE**: ~30 strikeouts
- **R²**: 0.36 (explains 36% of variance)
- **Dataset**: 989 pitcher-seasons (2021-2024 training) + 339 pitchers (2025 for predictions)
- **Features**: 59 engineered features
- **Latest Update**: January 2026

## 🎯 Model Details

### Features (59 total)
**Base Stats (13)**:
- Innings pitched, strikeouts, games, walks, pitches
- K/9, BB/9, HR/9, H/9, K/BB ratio, strike %
- ERA, WHIP, FIP

**Advanced Metrics (15)**:
- xFIP, SIERA, SwStr%, CSW%, Contact%
- O-Swing%, Z-Swing%, Zone%, F-Strike%
- Hard%, Barrel%, AVG against, LOB%
- Stuff+, Command+

**Age Features (7)**:
- Age, age², is_prime_age, is_young, is_veteran, age_from_peak

**Engineered Metrics (11)**:
- K-BB%, contact quality, whiff rate, zone contact diff
- True outcomes %, K to contact ratio, K upside
- Pitch efficiency, power index, consistency score

**Role Indicators (6)**:
- is_ace, is_high_k_pitcher, is_workhorse
- is_starter, is_reliever, log(strikeouts)

**Interactions (5)**:
- k9 × starter, swstr × starter, IP × starter
- swstr × IP, k9 × ERA

**Workload (2)**:
- age × IP, stuff × command, workload stress

### Algorithm
- **Ridge Regression** with L2 regularization (alpha=0.5)
- StandardScaler for feature normalization
- 80/20 train/test split

## 📁 Project Structure

```
strikeout_prediction/
├── README.md                          # This file
├── Predict_Pitcher_Strikeouts.ipynb  # Main analysis notebook
├── data/
│   ├── pitcher_season_averages_improved.csv      # Final training dataset (877 records)
│   ├── pitcher_season_averages_model_ready.csv   # Baseline (1,562 records)
│   └── pitcher_season_averages_with_*.csv        # Experimental datasets
├── scripts/
│   ├── create_improved_pitcher_dataset.py        # Feature engineering pipeline
│   ├── add_momentum_features.py                  # Year-over-year features (failed)
│   ├── add_pitch_mix_features.py                 # Pitch mix features (failed)
│   ├── separate_starter_reliever_models.py       # Split models (failed)
│   ├── test_tree_models.py                       # XGBoost/RF comparison (failed)
│   └── expand_to_2015_2025.py                    # Historical data expansion (failed)
├── tests/
│   ├── test_notebook_features.py                 # Validation tests
│   ├── test_momentum_model.py                    # Momentum feature tests
│   ├── test_lasso_feature_selection.py           # Feature selection tests
│   ├── test_pitch_mix_model.py                   # Pitch mix tests
│   ├── test_expanded_model.py                    # 2015-2025 data tests
│   ├── analyze_next_improvements.py              # Error analysis
│   └── analyze_remaining_options.py              # Improvement options analysis
└── models/
    └── (trained models can be saved here)
```

## 🧪 Experiments Conducted

| Approach | Result | Conclusion |
|----------|--------|------------|
| **Baseline** (44 features) | MAE 27.49 | Starting point |
| **Improved features** (59) | MAE 26.78 | ✅ **BEST** (+2.57%) |
| Momentum features (90) | MAE 28.44 | ❌ Worse (-6.2%) |
| Separate starter/reliever models | MAE 29.49 | ❌ Worse (-10.1%) |
| XGBoost | MAE 28.90 | ❌ Worse (-7.9%) |
| Random Forest | MAE 28.58 | ❌ Worse (-6.7%) |
| Gradient Boosting | MAE 29.06 | ❌ Worse (-8.5%) |
| Ensemble (weighted) | MAE 27.84 | ❌ Worse (-3.9%) |
| 2015-2025 data (3,988 records) | MAE 28.71 | ❌ Worse (-7.1%) |
| 2019-2025 data (2,317 records) | MAE 29.69 | ❌ Worse (-10.8%) |
| Lasso feature selection (14 features) | MAE 28.32 | ❌ Worse (-5.7%) |
| Pitch mix features (70 total) | MAE 30.29 | ❌ Worse (-13.1%) |

**Conclusion**: Ridge Regression with 59 carefully engineered features is optimal. All attempts to improve it failed.

## 🚀 Usage

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('data/pitcher_season_averages_improved.csv')

# Define features (see full list above)
feature_columns = [...]  # 59 features

X = df[feature_columns]
y = df['next_season_strikeouts']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = Ridge(alpha=0.5)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, R²: {r2:.4f}")
```

### Making Predictions

```python
# For a new pitcher season
new_pitcher_data = pd.DataFrame({
    'total_innings_pitched': [180.0],
    'total_strikeouts': [200],
    # ... all 59 features
})

# Scale and predict
new_pitcher_scaled = scaler.transform(new_pitcher_data)
predicted_strikeouts = model.predict(new_pitcher_scaled)[0]

print(f"Predicted next season strikeouts: {predicted_strikeouts:.0f}")
```

## 📖 Key Learnings

1. **Feature engineering > Complex algorithms**: Simple Ridge with good features beat XGBoost, Random Forest, etc.

2. **Quality > Quantity**: 877 recent records (2021-2025) beat 3,988 records spanning 2015-2025. Modern baseball data is more relevant.

3. **All features useful**: Lasso feature selection (keeping only 14 features) made performance worse, showing all 59 features contribute unique signal.

4. **Linear relationships dominate**: Strikeouts correlate linearly with K/9, SwStr%, and IP. Complex non-linear models overfit.

5. **Combined models > Separate**: Training separate models for starters/relievers performed worse than a single model with role features.

6. **External data not always helpful**: Adding pitch mix data (FB%, SL%, etc.) hurt performance because existing features already capture pitch quality indirectly through SwStr% and Stuff+.

## 🎓 Model Insights

**Top Predictive Features** (by importance):
1. `total_strikeouts` - Current season K count
2. `k_per_9` - Strikeout rate
3. `total_innings_pitched` - Workload/opportunity
4. `swstr_pct` - Swing & miss rate
5. `stuff_plus` - Pitch quality

**Why Ridge Wins**:
- Small dataset (877 records) relative to features (59)
- Strong linear relationships in strikeout data
- L2 regularization prevents overfitting better than tree pruning
- Feature engineering already captures non-linearity (interactions, log transforms)

## 🔮 Future Improvements

To get beyond MAE 26.78, you would need:

1. **More granular data**:
   - Pitch-by-pitch Statcast data
   - Health/injury indicators
   - Team context (park factors, defensive metrics)

2. **External factors**:
   - Contract year indicators
   - Role changes (starter → reliever)
   - Recovery from injury

3. **Neural networks** (if 1000+ samples):
   - Simple feedforward with dropout
   - Currently likely to overfit with 877 records

**Reality**: Further improvements have diminishing returns. Current MAE of ±27 strikeouts for next season is very good.

## 📊 Data Sources

- **Base stats**: FanGraphs via pybaseball
- **Statcast metrics**: Baseball Savant via pybaseball  
- **Seasons**: 2021-2025 (modern era)
- **Minimum threshold**: 50 IP per season

## 📝 License

MIT

## 👤 Author

Jack Holroyd
