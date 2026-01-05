# MLB Pitcher Strikeout Prediction

Machine learning model to predict next season strikeouts for MLB pitchers.

## 📊 Model Performance

**Final Model**: Random Forest Regressor (100 trees, max_depth=10)
- **MAE**: 28.66 strikeouts
- **R²**: 0.3566 (explains 35.7% of variance)
- **Dataset**: 989 pitcher-seasons (2021-2024 training) + 339 pitchers (2025 for predictions)
- **Features**: 59 engineered features
- **Latest Update**: January 2026
- **Improvement**: 4.2% better than Ridge Regression (29.92 MAE)

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
- **Random Forest Regressor** with 100 trees and max_depth=10
- No feature scaling required (tree-based model)
- Captures non-linear relationships better than linear models
- 4.2% improvement over Ridge Regression
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

We tested **multiple approaches** on the current dataset (989 records, 2021-2024):

| Approach | MAE | Result |
|----------|-----|--------|
| **Random Forest (100 trees, depth=10)** | **28.66 K** | **✅ BEST** |
| Ensemble (Ridge + RF + GB) | 29.26 K | -2.1% |
| Ridge Regression (59 features) | 29.92 K | -4.4% |
| Separate Starter/Reliever Models | 30.03 K | -4.8% |
| Gradient Boosting | 30.10 K | -5.0% |
| Lasso Feature Selection (13 features) | 30.31 K | -5.8% |

**Key Learnings:**
1. ✅ **Random Forest outperforms linear models** (MAE 28.66 vs 29.92)
2. ✅ Tree-based models better capture non-linear strikeout patterns
3. ✅ All 59 features contribute (Lasso selection worse)
4. ✅ Combined model > separate starter/reliever models
5. ✅ Ensemble close but RF alone is simpler and nearly as good
6. ✅ Recent data (2021-2024) optimal window

All experiment code available in `rerun_all_experiments.py`

## � 2026 Projections

**339 pitchers** qualified (50+ IP in 2025):
- **0 pitchers** projected for 200+ K (most conservative: 199 K)
- **23 pitchers** projected for 150+ K
- **129 pitchers** projected for 100+ K

**Top 5 Projected Leaders:**
1. Tarik Skubal (DET) - 199 K (range: 170-228)
2. Dylan Cease (SDP) - 190 K (range: 161-219)
3. Garrett Crochet (BOS) - 182 K (range: 153-211)
4. Jesus Luzardo (PHI) - 181 K (range: 152-210)
5. Zack Wheeler (PHI) - 178 K (range: 149-207)

**Model Insight**: Random Forest produces more conservative projections than linear models, with no pitchers breaking 200 K threshold.

## �🚀 Usage

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Train Random Forest (no scaling needed for tree-based models)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

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

# Predict (no scaling needed)
predicted_strikeouts = model.predict(new_pitcher_data)[0]

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

**Why Random Forest Wins**:
- Better captures non-linear interactions between features
- Handles feature interactions automatically (e.g., K/9 × IP × age)
- No feature scaling required
- More robust to outliers than linear models
- Tree splits naturally model conditional relationships (e.g., high K/9 matters more for starters)

## 🔮 Future Improvements

To get beyond MAE ~30, you would need:

1. **More granular data**:
   - Pitch-by-pitch Statcast data
   - Health/injury indicators
   - Team context (park factors, defensive metrics)

2. **External factors**:
   - Contract year indicators
   - Role changes (starter → reliever)
   - Recovery from injury

3. **Neural networks** (if dataset grows significantly):
   - Simple feedforward with dropout
   - Would need 2000+ records to avoid overfitting

**Reality**: Further improvements have diminishing returns. Current MAE of ±30 strikeouts is respectable for a full-season projection with public data.

## 📊 Data Sources

- **Base stats**: FanGraphs via pybaseball
- **Statcast metrics**: Baseball Savant via pybaseball  
- **Seasons**: 2021-2025 (modern era)
- **Minimum threshold**: 50 IP per season

## 📝 License

MIT

## 👤 Author

Jack Holroyd
