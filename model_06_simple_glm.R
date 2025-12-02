# Model 6: Simple GLM (Only VERIFIED_PURCHASE)
# This script trains a simple logistic regression model using only VERIFIED_PURCHASE

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
library(caret)
library(pROC)
library(ggplot2)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 6: Simple GLM (VERIFIED_PURCHASE only)\n")
cat("========================================\n\n")

# Train simple GLM model
cat("Training simple GLM model...\n")
only.verif <- glm(LABEL ~ VERIFIED_PURCHASE, data = reviews.train, family = 'binomial')

cat("Model training complete.\n")

# Make predictions
cat("Making predictions...\n")
only.verif.pred <- predict(only.verif, reviews.test, type = 'response')
class.only.verif <- rep("0", nrow(reviews.test))
class.only.verif[only.verif.pred > .5] <- "1"
class.only.verif <- factor(class.only.verif, levels = c("0", "1"))

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy
testacc.verif <- mean(class.only.verif == reviews.test$LABEL)
cat("Accuracy:", round(testacc.verif, 4), "\n")
cat("Error Rate:", round(1 - testacc.verif, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(class.only.verif, reviews.test$LABEL)
cat("\nConfusion Matrix:\n")
print(cm$table)

# Detailed metrics
cat("\nDetailed Metrics:\n")
cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(cm$byClass["F1"], 4), "\n")

# ROC Curve and AUC
roc_result <- roc(reviews.test$LABEL, only.verif.pred, quiet = TRUE)
cat("AUC:", round(as.numeric(auc(roc_result)), 4), "\n")

# Model Summary
cat("\nModel Coefficients:\n")
coef_summary <- summary(only.verif)$coefficients
print(coef_summary)

# Visualizations
cat("\nGenerating visualizations...\n")

# 0. Model Coefficients Visualization
coef_df <- data.frame(
  Variable = rownames(coef_summary),
  Coefficient = coef_summary[, "Estimate"],
  PValue = coef_summary[, "Pr(>|z|)"]
)
coef_df$Significant <- ifelse(coef_df$PValue < 0.05, "Significant", "Not Significant")
coef_df$Variable <- factor(coef_df$Variable, levels = coef_df$Variable[order(abs(coef_df$Coefficient))])

ggplot(coef_df, aes(x = Coefficient, y = Variable, fill = Significant)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Significant" = "darkgreen", "Not Significant" = "lightgray")) +
  labs(title = "Model Coefficients - Simple GLM (VERIFIED_PURCHASE only)",
       x = "Coefficient Value", y = "Variable", fill = "Significance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black")
ggsave("figures/model_06_coefficients.png", width = 8, height = 4, dpi = 300)

# 1. ROC Curve
png("figures/model_06_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - Simple GLM (VERIFIED_PURCHASE only)", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 2. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Simple GLM",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_06_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 3. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = only.verif.pred,
  Label = reviews.test$LABEL
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - Simple GLM",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_06_probability_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_06_coefficients.png\n")
cat("  - model_06_roc_curve.png\n")
cat("  - model_06_confusion_matrix.png\n")
cat("  - model_06_probability_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

