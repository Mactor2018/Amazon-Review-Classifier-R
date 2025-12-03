# Model 4: Ranger (Fast Random Forest)
# This script trains a Ranger model for fake review classification

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
library(ranger)
library(caret)
library(pROC)
library(ggplot2)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 4: Ranger (Fast Random Forest)\n")
cat("========================================\n\n")

# Train Ranger model with probability=TRUE to get proper probability predictions
cat("Training Ranger model...\n")
cat(sprintf("Training set size: %d samples\n", nrow(reviews.train)))

# 大数据集优化参数:
# - num.trees: 100 (减少树数量)
# - min.node.size: 20 (增大叶节点最小样本数)
# - sample.fraction: 0.5 (子采样比例，加速训练)
# - num.threads: 并行线程数

reviews.rf <- ranger(LABEL ~ . - DOC_ID - PRODUCT_ID - PRODUCT_TITLE - REVIEW_TITLE - REVIEW_TEXT, 
                     reviews.train, 
                     num.trees = 100, 
                     min.node.size = 20,
                     sample.fraction = 0.5,
                     importance = "impurity",
                     probability = TRUE,
                     num.threads = parallel::detectCores() - 1)

cat("Model training complete.\n")
cat("Out-of-bag prediction error:", round(reviews.rf$prediction.error, 4), "\n")

# Make predictions
cat("Making predictions...\n")
predict.rf <- predict(reviews.rf, data = reviews.test, type = "response")

# Extract probability for class "1" (fake review)
predict.rf.prob <- predict.rf$predictions[, "1"]

# Convert probabilities to class predictions (threshold = 0.5)
predict.rf.class <- factor(ifelse(predict.rf.prob > 0.5, "1", "0"), levels = c("0", "1"))

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy
testacc.rf <- mean(reviews.test$LABEL == predict.rf.class)
cat("Accuracy:", round(testacc.rf, 4), "\n")
cat("Error Rate:", round(1 - testacc.rf, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(predict.rf.class, reviews.test$LABEL)
cat("\nConfusion Matrix:\n")
print(cm$table)

# Detailed metrics
cat("\nDetailed Metrics:\n")
cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(cm$byClass["F1"], 4), "\n")

# ROC Curve and AUC
roc_result <- roc(reviews.test$LABEL, predict.rf.prob, quiet = TRUE)
cat("AUC:", round(as.numeric(auc(roc_result)), 4), "\n")

# Variable Importance
cat("\nTop 30 Most Important Variables:\n")
imp_sorted <- reviews.rf$variable.importance[order(reviews.rf$variable.importance, decreasing = TRUE)]
print(head(imp_sorted, 30))

# Visualizations
cat("\nGenerating visualizations...\n")

# 1. Variable Importance Plot
imp_top30 <- head(imp_sorted, 30)
imp_df <- data.frame(
  Variable = names(imp_top30),
  Importance = as.numeric(imp_top30)
)
imp_df$Variable <- factor(imp_df$Variable, levels = imp_df$Variable[order(imp_df$Importance)])

ggplot(imp_df, aes(x = Importance, y = Variable)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  labs(title = "Top 30 Variable Importance - Ranger",
       x = "Importance (Impurity)", y = "Variable") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_04_variable_importance.png", width = 10, height = 10, dpi = 300)

# 2. ROC Curve
png("figures/model_04_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - Ranger", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 3. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Ranger",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_04_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 4. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = predict.rf.prob,
  Label = reviews.test$LABEL
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - Ranger",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_04_prediction_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_04_variable_importance.png\n")
cat("  - model_04_roc_curve.png\n")
cat("  - model_04_confusion_matrix.png\n")
cat("  - model_04_prediction_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

