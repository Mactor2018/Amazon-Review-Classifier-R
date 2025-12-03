# Model 3: Random Forest
# This script trains a Random Forest model for fake review classification

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 3: Random Forest\n")
cat("========================================\n\n")

# Train Random Forest model
cat("Training Random Forest model...\n")
cat("This may take a few minutes for large datasets...\n")

# 大数据集优化参数:
# - ntree: 50 (减少树数量，50 通常足够)
# - nodesize: 20 (增大叶节点最小样本数，显著加速训练)
# - mtry: 30 (减少每次分裂考虑的特征数)
# - sampsize: 子采样大小，避免内存问题
# - importance: TRUE (需要特征重要性)

n_train <- nrow(reviews.train)
sample_size <- min(n_train, 5000)  # 最多使用 5000 样本进行训练
cat(sprintf("Training set: %d samples, subsample: %d\n", n_train, sample_size))

rf.model <- randomForest(LABEL ~ . - DOC_ID - PRODUCT_ID - PRODUCT_TITLE - REVIEW_TITLE - REVIEW_TEXT, 
                        reviews.train, 
                        ntree = 50,
                        nodesize = 20,
                        mtry = 30,
                        sampsize = sample_size,
                        importance = TRUE)

cat("Model training complete.\n")

# Make predictions
cat("Making predictions...\n")
predict.rf <- predict(rf.model, reviews.test, type = "response")
predict.rf.prob <- predict(rf.model, reviews.test, type = "prob")[,2]

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy
testacc.rf <- mean(reviews.test$LABEL == predict.rf)
cat("Accuracy:", round(testacc.rf, 4), "\n")
cat("Error Rate:", round(1 - testacc.rf, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(predict.rf, reviews.test$LABEL)
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
cat("\nTop 20 Most Important Variables:\n")
imp <- importance(rf.model)
imp_sorted <- imp[order(imp[, "MeanDecreaseGini"], decreasing = TRUE), ]
print(head(imp_sorted, 20))

# Visualizations
cat("\nGenerating visualizations...\n")

# 1. Error Rate Plot
png("figures/model_03_rf_error_rate.png", width = 800, height = 600)
plot(rf.model, main = "Random Forest Error Rate")
legend("topright", legend = c("OOB", "Real", "Fake"), 
       col = c("black", "green", "red"), lty = 1)
dev.off()

# 2. Variable Importance Plot
imp_top20 <- head(imp_sorted, 20)
imp_df <- data.frame(
  Variable = rownames(imp_top20),
  Importance = imp_top20[, "MeanDecreaseGini"]
)
imp_df$Variable <- factor(imp_df$Variable, levels = imp_df$Variable[order(imp_df$Importance)])

ggplot(imp_df, aes(x = Importance, y = Variable)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Top 20 Variable Importance - Random Forest",
       x = "Mean Decrease Gini", y = "Variable") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_03_variable_importance.png", width = 10, height = 8, dpi = 300)

# 3. ROC Curve
png("figures/model_03_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - Random Forest", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 4. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Random Forest",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_03_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 5. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = predict.rf.prob,
  Label = reviews.test$LABEL
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - Random Forest",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_03_probability_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_03_rf_error_rate.png\n")
cat("  - model_03_variable_importance.png\n")
cat("  - model_03_roc_curve.png\n")
cat("  - model_03_confusion_matrix.png\n")
cat("  - model_03_probability_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

