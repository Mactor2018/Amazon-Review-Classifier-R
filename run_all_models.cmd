@echo off
REM ============================================
REM Run All Models Script
REM This script runs data preprocessing and all model scripts
REM Outputs are saved to logs directory
REM ============================================

echo ============================================
echo Amazon Review Classifier - Batch Run
echo ============================================
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Get timestamp for log files
set timestamp=%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%

echo Starting batch run at %date% %time%
echo Log files will be saved to logs\ directory
echo.

REM Step 1: Data Preprocessing
echo [1/9] Running data preprocessing...
Rscript data_preprocessing.R > logs\01_data_preprocessing_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Data preprocessing failed! Check logs\01_data_preprocessing_%timestamp%.log
    pause
    exit /b 1
)
echo Data preprocessing completed successfully.
echo.

REM Step 2: Model 01 - GLM Lasso Min
echo [2/9] Running Model 01 - GLM Lasso (lambda.min)...
Rscript model_01_glm_lasso_min.R > logs\02_model_01_glm_lasso_min_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 01 failed! Check logs\02_model_01_glm_lasso_min_%timestamp%.log
) else (
    echo Model 01 completed successfully.
)
echo.

REM Step 3: Model 02 - GLM Lasso 1se
echo [3/9] Running Model 02 - GLM Lasso (lambda.1se)...
Rscript model_02_glm_lasso_1se.R > logs\03_model_02_glm_lasso_1se_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 02 failed! Check logs\03_model_02_glm_lasso_1se_%timestamp%.log
) else (
    echo Model 02 completed successfully.
)
echo.

REM Step 4: Model 03 - Random Forest
echo [4/9] Running Model 03 - Random Forest...
Rscript model_03_random_forest.R > logs\04_model_03_random_forest_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 03 failed! Check logs\04_model_03_random_forest_%timestamp%.log
) else (
    echo Model 03 completed successfully.
)
echo.

REM Step 5: Model 04 - Ranger
echo [5/9] Running Model 04 - Ranger...
Rscript model_04_ranger.R > logs\05_model_04_ranger_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 04 failed! Check logs\05_model_04_ranger_%timestamp%.log
) else (
    echo Model 04 completed successfully.
)
echo.

REM Step 6: Model 05 - XGBoost
echo [6/9] Running Model 05 - XGBoost...
Rscript model_05_xgboost.R > logs\06_model_05_xgboost_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 05 failed! Check logs\06_model_05_xgboost_%timestamp%.log
) else (
    echo Model 05 completed successfully.
)
echo.

REM Step 7: Model 06 - Simple GLM
echo [7/9] Running Model 06 - Simple GLM...
Rscript model_06_simple_glm.R > logs\07_model_06_simple_glm_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 06 failed! Check logs\07_model_06_simple_glm_%timestamp%.log
) else (
    echo Model 06 completed successfully.
)
echo.

REM Step 8: Model 07 - Neural Network (REVIEW_TEXT only)
echo [8/9] Running Model 07 - Neural Network (REVIEW_TEXT only)...
echo This may take several minutes...
Rscript model_07_neural_network.R > logs\08_model_07_neural_network_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 07 failed! Check logs\08_model_07_neural_network_%timestamp%.log
    echo Note: Neural network models require keras3 package. See INSTALL_NEURAL_NETWORK.md
) else (
    echo Model 07 completed successfully.
)
echo.

REM Step 9: Model 08 - Neural Network (REVIEW_TEXT + REVIEW_TITLE)
echo [9/9] Running Model 08 - Neural Network (REVIEW_TEXT + REVIEW_TITLE)...
echo This may take several minutes...
Rscript model_08_neural_network_titles.R > logs\09_model_08_neural_network_titles_%timestamp%.log 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Model 08 failed! Check logs\09_model_08_neural_network_titles_%timestamp%.log
    echo Note: Neural network models require keras3 package. See INSTALL_NEURAL_NETWORK.md
) else (
    echo Model 08 completed successfully.
)
echo.

REM Summary
echo ============================================
echo Batch Run Complete!
echo ============================================
echo.
echo All outputs have been saved to logs\ directory:
echo   - Data preprocessing: logs\01_data_preprocessing_%timestamp%.log
echo   - Model 01: logs\02_model_01_glm_lasso_min_%timestamp%.log
echo   - Model 02: logs\03_model_02_glm_lasso_1se_%timestamp%.log
echo   - Model 03: logs\04_model_03_random_forest_%timestamp%.log
echo   - Model 04: logs\05_model_04_ranger_%timestamp%.log
echo   - Model 05: logs\06_model_05_xgboost_%timestamp%.log
echo   - Model 06: logs\07_model_06_simple_glm_%timestamp%.log
echo   - Model 07: logs\08_model_07_neural_network_%timestamp%.log
echo   - Model 08: logs\09_model_08_neural_network_titles_%timestamp%.log
echo.
echo All visualizations have been saved to figures\ directory.
echo.
echo Check the log files for detailed output and any errors.
echo.

pause

