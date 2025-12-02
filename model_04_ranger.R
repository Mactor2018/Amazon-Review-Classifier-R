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

# Train Ranger model
cat("Training Ranger model...\n")
reviews.rf <- ranger(LABEL ~ . - DOC_ID - PRODUCT_ID - PRODUCT_TITLE - REVIEW_TITLE - REVIEW_TEXT, 
                     reviews.train, 
                     num.trees = 200, 
                     importance = "impurity")

cat("Model training complete.\n")
cat("Out-of-bag prediction error:", round(reviews.rf$prediction.error, 4), "\n")

# Make predictions
cat("Making predictions...\n")
predict.rf <- predict(reviews.rf, data = reviews.test, type = "response")
predict.rf.prob <- predict(reviews.rf, data = reviews.test, type = "response")$predictions

# Convert predictions to factor
predict.rf.class <- factor(predict.rf$predictions, levels = c("0", "1"))

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
# Note: Ranger doesn't provide probability predictions directly, so we use predictions as proxy
roc_result <- roc(reviews.test$LABEL, as.numeric(as.character(predict.rf.class)), quiet = TRUE)
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

# 4. Prediction Probability Distribution (using predictions as proxy)
pred_df <- data.frame(
  Prediction = as.numeric(as.character(predict.rf.class)),
  Label = as.numeric(as.character(reviews.test$LABEL))
)
ggplot(pred_df, aes(x = Prediction, fill = factor(Label))) +
  geom_bar(alpha = 0.7, position = "dodge") +
  labs(title = "Prediction Distribution - Ranger",
       x = "Predicted Class", y = "Count", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  scale_x_continuous(breaks = c(0, 1), labels = c("Real (0)", "Fake (1)"))
ggsave("figures/model_04_prediction_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_04_variable_importance.png\n")
cat("  - model_04_roc_curve.png\n")
cat("  - model_04_confusion_matrix.png\n")
cat("  - model_04_prediction_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

