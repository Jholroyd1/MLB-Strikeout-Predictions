#!/bin/bash

echo "=================================================="
echo "CLEANING UP STRIKEOUT PREDICTION FOLDER"
echo "=================================================="

cd /Users/jackholroyd/MLB_Stats/strikeout_prediction

# Create archive for failed experiments
echo ""
echo "📦 Creating archive folder for failed experiments..."
mkdir -p archive/failed_experiments
mkdir -p archive/old_datasets

# Move failed experiment files
echo ""
echo "🗑️  Archiving failed experiments..."

# Failed datasets (from experiments that didn't improve model)
mv data/pitcher_season_averages_model_ready.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_enhanced_qualified.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_enhanced.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_improved_2015_2025.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_age.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_arsenal.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_injury_features.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_momentum.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_pitch_mix.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages_with_statcast.csv archive/old_datasets/ 2>/dev/null
mv data/pitcher_season_averages.csv archive/old_datasets/ 2>/dev/null

# Failed experiment scripts
mv scripts/add_injury_features.py archive/failed_experiments/ 2>/dev/null
mv scripts/add_momentum_features.py archive/failed_experiments/ 2>/dev/null
mv scripts/add_pitch_mix_features.py archive/failed_experiments/ 2>/dev/null
mv scripts/clean_expanded_dataset.py archive/failed_experiments/ 2>/dev/null
mv scripts/expand_to_2015_2025.py archive/failed_experiments/ 2>/dev/null
mv scripts/separate_starter_reliever_models.py archive/failed_experiments/ 2>/dev/null
mv scripts/test_tree_models.py archive/failed_experiments/ 2>/dev/null

# Test files (keep for reference but archive)
mv tests/*.py archive/failed_experiments/ 2>/dev/null

# Failed models
mv models/injury_enhanced_predictor.pkl archive/failed_experiments/ 2>/dev/null
mv models/separate_models.pkl archive/failed_experiments/ 2>/dev/null

# Outdated docs
mv INJURY_MODEL_SUMMARY.md archive/failed_experiments/ 2>/dev/null
mv predict_with_injuries.py archive/failed_experiments/ 2>/dev/null

echo "   ✓ Failed experiments archived"

# Remove empty tests directory if it exists
rmdir tests 2>/dev/null

# Keep only essential files
echo ""
echo "✅ KEEPING ESSENTIAL FILES:"
echo ""
echo "📊 Data:"
echo "   ✓ data/pitcher_season_averages_improved.csv (training data 2021-2024)"
echo "   ✓ data/pitcher_season_averages_improved_full.csv (includes 2025 for predictions)"
echo "   ✓ data/2026_strikeout_projections.csv (2026 predictions)"
echo ""
echo "📜 Scripts:"
echo "   ✓ scripts/update_to_latest_data.py (data collection & feature engineering)"
echo ""
echo "📓 Documentation:"
echo "   ✓ README.md (project documentation)"
echo "   ✓ QUICK_START.md (usage guide)"
echo ""
echo "🪐 Notebook:"
echo "   ✓ Predict_Pitcher_Strikeouts.ipynb (interactive analysis)"
echo ""
echo "📦 Archived:"
echo "   ✓ archive/failed_experiments/ (12+ failed improvement attempts)"
echo "   ✓ archive/old_datasets/ (11 experimental datasets)"

echo ""
echo "=================================================="
echo "✅ CLEANUP COMPLETE!"
echo "=================================================="
echo ""
echo "Final structure:"
find . -maxdepth 2 -type f \( -name "*.py" -o -name "*.csv" -o -name "*.ipynb" -o -name "*.md" \) | grep -v archive | grep -v __pycache__ | sort

