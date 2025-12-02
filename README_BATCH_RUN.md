# Batch Run Script

This directory contains a batch script to run all models automatically.

## Quick Start

Simply double-click `run_all_models.bat` or `run_all_models.cmd`, or run from command line:

```cmd
run_all_models.bat
```

## What It Does

The script will:

1. **Create logs directory** (if it doesn't exist)
2. **Run data preprocessing** (`data_preprocessing.R`)
3. **Run all 8 models** sequentially:
   - Model 01: GLM Lasso (lambda.min)
   - Model 02: GLM Lasso (lambda.1se)
   - Model 03: Random Forest
   - Model 04: Ranger
   - Model 05: XGBoost
   - Model 06: Simple GLM
   - Model 07: Neural Network (REVIEW_TEXT only)
   - Model 08: Neural Network (REVIEW_TEXT + REVIEW_TITLE)

## Output Files

### Log Files
All console output is saved to the `logs/` directory with timestamped filenames:
- `01_data_preprocessing_YYYYMMDD_HHMMSS.log`
- `02_model_01_glm_lasso_min_YYYYMMDD_HHMMSS.log`
- `03_model_02_glm_lasso_1se_YYYYMMDD_HHMMSS.log`
- `04_model_03_random_forest_YYYYMMDD_HHMMSS.log`
- `05_model_04_ranger_YYYYMMDD_HHMMSS.log`
- `06_model_05_xgboost_YYYYMMDD_HHMMSS.log`
- `07_model_06_simple_glm_YYYYMMDD_HHMMSS.log`
- `08_model_07_neural_network_YYYYMMDD_HHMMSS.log`
- `09_model_08_neural_network_titles_YYYYMMDD_HHMMSS.log`

### Visualizations
All model visualizations are automatically saved to the `figures/` directory.

### Model Files
Trained models are saved to the root directory:
- `reviews.train.RData` (training data)
- `rf.mtry.RData` (Random Forest model)
- `reviews.rf.RData` (Ranger model)
- `fakereview.keras` (Neural Network model 07)
- `fakereview_titles.keras` (Neural Network model 08)

## Error Handling

- If data preprocessing fails, the script will stop and show an error message
- If individual models fail, the script will continue with remaining models and show warnings
- Check the corresponding log files for detailed error messages

## Notes

- **Neural Network Models**: Models 07 and 08 require `keras3` package. If not installed, they will fail gracefully. See `INSTALL_NEURAL_NETWORK.md` for installation instructions.
- **Execution Time**: The full batch run may take 30-60 minutes depending on your system:
  - Models 01-06: ~5-15 minutes total
  - Model 07: ~10-20 minutes
  - Model 08: ~15-30 minutes
- **Memory**: Ensure you have sufficient RAM (recommended: 8GB+)

## Running Individual Models

If you only want to run specific models, you can run them individually:

```cmd
Rscript model_01_glm_lasso_min.R > logs\model_01.log 2>&1
```

Or in R:
```r
source('model_01_glm_lasso_min.R')
```

## Troubleshooting

### Script doesn't run
- Make sure R is installed and `Rscript` is in your PATH
- Check that all R script files exist in the current directory

### Models fail
- Check the corresponding log file in `logs/` directory
- Ensure all required R packages are installed (see `README_MODELS.md`)
- For neural network models, ensure `keras3` is installed (see `INSTALL_NEURAL_NETWORK.md`)

### Log files are empty
- Check if Rscript is working: `Rscript --version`
- Check file permissions in the logs directory

