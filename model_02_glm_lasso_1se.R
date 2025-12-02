# Model 2: GLM with Lasso Regularization (lambda.1se)
# This script trains a logistic regression model with Lasso regularization
# using lambda.1se for feature selection (more conservative)

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
library(Matrix)
library(glmnet)
library(pROC)
library(caret)
library(ggplot2)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 2: GLM with Lasso (lambda.1se)\n")
cat("========================================\n\n")

# Prepare data for Lasso
cat("Preparing data for Lasso regression...\n")
X <- sparse.model.matrix(LABEL ~ . - DOC_ID - PRODUCT_ID - PRODUCT_TITLE - REVIEW_TITLE - REVIEW_TEXT, 
                         data = reviews.train)[,-1]
Y <- reviews.train$LABEL

# Cross-validation for Lasso
cat("Running cross-validation for Lasso...\n")
reviews.lasso <- cv.glmnet(X, Y, family = "binomial")

# Get coefficients at lambda.1se
beta.lasso.1se <- coef(reviews.lasso, s="lambda.1se")
beta <- beta.lasso.1se[which(beta.lasso.1se != 0),]
beta <- as.matrix(beta)
beta <- rownames(beta)

cat("Number of selected features:", length(beta) - 1, "\n")  # -1 for intercept

# Build GLM formula with selected features
glm.input.1se <- as.formula(paste("LABEL", "~ VERIFIED_PURCHASE + PRODUCT_CATEGORY + RATING +", 
                                   paste(beta[-(1:14)], collapse = "+")))

# Train GLM model
cat("Training GLM model...\n")
reviews.glm.1se <- glm(glm.input.1se, family=binomial, reviews.train)

# Make predictions
cat("Making predictions...\n")
predict.glm.1se <- predict(reviews.glm.1se, reviews.test, type = "response")
class.glm.1se <- rep("0", nrow(reviews.test))
class.glm.1se[predict.glm.1se > .5] <- "1"
class.glm.1se <- factor(class.glm.1se, levels = c("0", "1"))

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy
testacc.glm.1se <- mean(reviews.test$LABEL == class.glm.1se)
cat("Accuracy:", round(testacc.glm.1se, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(class.glm.1se, reviews.test$LABEL)
cat("\nConfusion Matrix:\n")
print(cm$table)

# Detailed metrics
cat("\nDetailed Metrics:\n")
cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(cm$byClass["F1"], 4), "\n")

# ROC Curve and AUC
roc_result <- roc(reviews.test$LABEL, predict.glm.1se, quiet = TRUE)
cat("AUC:", round(as.numeric(auc(roc_result)), 4), "\n")

# Visualizations
cat("\nGenerating visualizations...\n")

# 1. Lasso Cross-Validation Plot
png("figures/model_02_lasso_cv_plot.png", width = 800, height = 600)
plot(reviews.lasso)
title("Lasso Cross-Validation (lambda.1se)")
dev.off()

# 2. ROC Curve
png("figures/model_02_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - GLM with Lasso (lambda.1se)", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 3. Lasso Coefficient Path
png("figures/model_02_lasso_coefficient_path.png", width = 1000, height = 600)
plot(reviews.lasso$glmnet.fit, xvar = "lambda", label = FALSE)
abline(v = log(reviews.lasso$lambda.1se), lty = 2, col = "blue", lwd = 2)
title("Lasso Coefficient Path (lambda.1se)")
dev.off()

# 4. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - GLM with Lasso (lambda.1se)",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_02_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 5. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = predict.glm.1se,
  Label = reviews.test$LABEL
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - GLM Lasso (lambda.1se)",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_02_probability_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_02_lasso_cv_plot.png\n")
cat("  - model_02_lasso_coefficient_path.png\n")
cat("  - model_02_roc_curve.png\n")
cat("  - model_02_confusion_matrix.png\n")
cat("  - model_02_probability_distribution.png\n")

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

