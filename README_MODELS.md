# Model Scripts for Amazon Review Classifier

This directory contains individual R scripts for each machine learning model used to classify fake Amazon reviews.

## Prerequisites

Make sure you have installed all required R packages:

```r
install.packages(c("tm", "magrittr", "SnowballC", "Matrix", "glmnet", "pROC", "caret", 
                   "randomForest", "ranger", "xgboost", "keras3", "ade4", 
                   "dplyr", "ggplot2", "gridExtra"))
```

For keras3 (required for neural network models 7-8):

**Step 1: Install keras3 package**
```r
install.packages("keras3")
```

If CRAN installation fails, try installing from GitHub:
```r
install.packages("remotes")
remotes::install_github("rstudio/keras3")
```

**Step 2: TensorFlow backend installation**

After installing keras3, TensorFlow will be automatically installed when you first run:
```r
library(keras3)
use_backend("tensorflow")
```

This happens automatically when you run the neural network model scripts.

**Note**: 
- The `install_keras()` function is not available in keras3
- TensorFlow installation is handled automatically when you first use `use_backend("tensorflow")`
- If you encounter issues, see `INSTALL_NEURAL_NETWORK.md` for detailed troubleshooting steps

## Data File

All scripts expect `data_new.csv` to be in the same directory. This file contains the preprocessed Amazon reviews data.

## Model Scripts

### 1. `model_01_glm_lasso_min.R`
- **Model**: Logistic Regression with Lasso Regularization (lambda.min)
- **Features**: Selected features from Lasso regularization
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC

### 2. `model_02_glm_lasso_1se.R`
- **Model**: Logistic Regression with Lasso Regularization (lambda.1se)
- **Features**: More conservative feature selection
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC

### 3. `model_03_random_forest.R`
- **Model**: Random Forest
- **Parameters**: ntree=100, nodesize=5, mtry=60
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC, Variable Importance

### 4. `model_04_ranger.R`
- **Model**: Ranger (Fast Random Forest)
- **Parameters**: num.trees=200
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC, Variable Importance

### 5. `model_05_xgboost.R`
- **Model**: XGBoost (Gradient Boosting)
- **Parameters**: eta=0.05, nrounds=342
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC, Feature Importance

### 6. `model_06_simple_glm.R`
- **Model**: Simple Logistic Regression
- **Features**: Only VERIFIED_PURCHASE
- **Output**: Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC, Model Coefficients

### 7. `model_07_neural_network.R`
- **Model**: Neural Network (Keras3)
- **Features**: Review text features only
- **Architecture**: 2 hidden layers (32-32 units), dropout=0.3
- **Output**: Test Loss, Test Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC
- **Saves**: Model to `fakereview.keras`

### 8. `model_08_neural_network_titles.R`
- **Model**: Neural Network (Keras3)
- **Features**: Review text + Review title features
- **Architecture**: 3 hidden layers (32-32-16 units), dropout=0.3
- **Output**: Test Loss, Test Accuracy, Confusion Matrix, Sensitivity, Specificity, Precision, F1 Score, AUC
- **Saves**: Model to `fakereview_titles.keras`

## Usage

To run any model, simply execute the R script:

```r
source('model_01_glm_lasso_min.R')
```

Or from command line:

```bash
Rscript model_01_glm_lasso_min.R
```

## Shared Data Preprocessing

All models use `data_preprocessing.R` which:
- Loads the data from `data_new.csv`
- Preprocesses text data (removes stopwords, punctuation, numbers, stems)
- Creates document-term matrix
- Splits data into training (75%) and test (25%) sets

## Performance Metrics

Each model outputs:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: True/False Positives and Negatives
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Precision**: Positive Predictive Value
- **F1 Score**: Harmonic mean of Precision and Recall
- **AUC**: Area Under the ROC Curve

## Visualizations

All models automatically generate and save visualizations to the `figures/` directory:

- **ROC Curves**: Receiver Operating Characteristic curves for all models
- **Confusion Matrix Heatmaps**: Visual representation of classification results
- **Model-Specific Visualizations**:
  - **Lasso models** (models 1-2): Cross-validation plots showing lambda selection
  - **Random Forest** (model 3): Error rate plot and variable importance
  - **Ranger** (model 4): Variable importance plot
  - **XGBoost** (model 5): Feature importance plot
  - **Simple GLM** (model 6): Prediction probability distribution
  - **Neural Networks** (models 7-8): Training history plots (loss and accuracy) and prediction probability distributions

All figures are saved as high-resolution PNG files (300 DPI) with descriptive filenames following the pattern `model_XX_description.png`.

## Notes

- All models use the same random seed (245) for reproducibility
- Training/test split is 75/25
- Neural network models may take several minutes to train
- Some models save their trained models to disk for later use
- All visualizations are automatically saved to the `figures/` directory when you run each model script
- The `figures/` directory will be created automatically if it doesn't exist

