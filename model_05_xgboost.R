# Model 5: XGBoost (Gradient Boosting)
# This script trains an XGBoost model for fake review classification

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
library(xgboost)
library(caret)
library(pROC)
library(ggplot2)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 5: XGBoost (Gradient Boosting)\n")
cat("========================================\n\n")

# Prepare data
cat("Preparing data for XGBoost...\n")
boost.data <- reviews.train[,-c(1,2,6,7,8,9)]
boost.data$PRODUCT_CATEGORY <- as.numeric(boost.data$PRODUCT_CATEGORY)
boost.data$VERIFIED_PURCHASE <- as.numeric(boost.data$VERIFIED_PURCHASE) - 1

boost.test <- reviews.test[,-c(1,2,6,7,8,9)]
boost.test$VERIFIED_PURCHASE <- as.numeric(boost.test$VERIFIED_PURCHASE) - 1
boost.test$PRODUCT_CATEGORY <- as.numeric(boost.test$PRODUCT_CATEGORY)

# Create DMatrix
boost.train <- xgb.DMatrix(data = as.matrix(boost.data[,-1]), 
                          label = as.numeric(as.character(reviews.train$LABEL)))

# Train XGBoost model
cat("Training XGBoost model...\n")
cat("Parameters: eta = 0.05, nrounds = 342\n")

# Train model with evaluation history tracking
bstDMatrix <- xgb.train(data = boost.train, 
                        eta = 0.05, 
                        nrounds = 342,
                        objective = "binary:logistic",
                        eval_metric = "error",
                        watchlist = list(train = boost.train),
                        verbose = 0,
                        save_period = 0)

cat("Model training complete.\n")

# Make predictions
cat("Making predictions...\n")
pred.boost <- predict(bstDMatrix, as.matrix(boost.test[,-1]))
class.boost <- rep("0", nrow(boost.test))
class.boost[pred.boost > .5] <- "1"
class.boost <- factor(class.boost, levels = c("0", "1"))

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy
testacc.boost <- mean(class.boost == reviews.test$LABEL)
cat("Accuracy:", round(testacc.boost, 4), "\n")
cat("Error Rate:", round(1 - testacc.boost, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(class.boost, reviews.test$LABEL)
cat("\nConfusion Matrix:\n")
print(cm$table)

# Detailed metrics
cat("\nDetailed Metrics:\n")
cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(cm$byClass["F1"], 4), "\n")

# ROC Curve and AUC
roc_result <- roc(reviews.test$LABEL, pred.boost, quiet = TRUE)
cat("AUC:", round(as.numeric(auc(roc_result)), 4), "\n")

# Feature Importance
cat("\nTop 20 Most Important Features:\n")
mat <- xgb.importance(feature_names = colnames(boost.data[,-1]), model = bstDMatrix)
print(head(mat, 20))

# Visualizations
cat("\nGenerating visualizations...\n")

# 0. Training Error History (if available)
eval_log <- bstDMatrix$evaluation_log
if (!is.null(eval_log) && nrow(eval_log) > 0 && "train_error" %in% colnames(eval_log)) {
  eval_df <- data.frame(
    Iteration = 1:nrow(eval_log),
    Train_Error = eval_log$train_error
  )
  ggplot(eval_df, aes(x = Iteration, y = Train_Error)) +
    geom_line(color = "steelblue", size = 1) +
    labs(title = "XGBoost Training Error History",
         x = "Iteration (Boosting Round)", y = "Training Error") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  ggsave("figures/model_05_training_error.png", width = 10, height = 6, dpi = 300)
  cat("  - model_05_training_error.png\n")
}

# 1. Feature Importance Plot
mat_top20 <- head(mat, 20)
png("figures/model_05_feature_importance.png", width = 1000, height = 800)
xgb.plot.importance(importance_matrix = mat_top20, 
                    main = "Top 20 Feature Importance - XGBoost")
dev.off()

# 2. ROC Curve
png("figures/model_05_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - XGBoost", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 3. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - XGBoost",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_05_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 4. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = pred.boost,
  Label = reviews.test$LABEL
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - XGBoost",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_05_probability_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
eval_log <- bstDMatrix$evaluation_log
if (!is.null(eval_log) && nrow(eval_log) > 0 && "train_error" %in% colnames(eval_log)) {
  cat("  - model_05_training_error.png\n")
}
cat("  - model_05_feature_importance.png\n")
cat("  - model_05_roc_curve.png\n")
cat("  - model_05_confusion_matrix.png\n")
cat("  - model_05_probability_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

