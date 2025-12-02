# Model 7: Neural Network (REVIEW_TEXT only)
# This script trains a neural network model using keras3

# Source data preprocessing
source('data_preprocessing.R')

# Load required libraries
# Check if keras3 is installed
if (!require("keras3", quietly = TRUE)) {
  cat("ERROR: keras3 package is not installed.\n")
  cat("Please install it using one of the following methods:\n")
  cat("1. From CRAN: install.packages('keras3')\n")
  cat("2. From GitHub: remotes::install_github('rstudio/keras3')\n")
  cat("\nAfter installation, TensorFlow will be automatically installed when you run use_backend('tensorflow')\n")
  stop("keras3 package is required but not installed.")
}

library(keras3)
library(ade4)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

cat("\n========================================\n")
cat("Model 7: Neural Network (REVIEW_TEXT only)\n")
cat("========================================\n\n")

# Set backend
use_backend("tensorflow")

# Prepare neural network data
cat("Preparing data for neural network...\n")
nn.data <- reviews.corpus[, -c(1,6,7,8,9)]

# One-hot encode product category
dummy <- acm.disjonctif(nn.data['PRODUCT_CATEGORY'])
nn.data['PRODUCT_CATEGORY'] = NULL
nn.data <- cbind(nn.data, dummy)

# Recode VERIFIED_PURCHASE
nn.data$VERIFIED_PURCHASE <- as.numeric(nn.data$VERIFIED_PURCHASE) - 1

# Split data
set.seed(245)
train.indices <- sample(nrow(nn.data), 0.75*nrow(nn.data))
nn.data.test <- nn.data[-train.indices, ]
nn.data.train <- nn.data[train.indices, ]

# Prepare training and test data
cat("Converting data to numeric matrices...\n")
nn.data.train.x <- nn.data.train[, -1]
nn.data.train.x <- as.data.frame(lapply(nn.data.train.x, function(x) {
  if (is.factor(x)) as.numeric(as.character(x))
  else if (is.character(x)) as.numeric(x)
  else as.numeric(x)
}))
nn.data.train.x <- as.matrix(nn.data.train.x)
storage.mode(nn.data.train.x) <- "numeric"

nn.data.train.y <- as.numeric(as.character(nn.data.train[, 1]))
nn.data.train.y <- matrix(nn.data.train.y, ncol = 1)
storage.mode(nn.data.train.y) <- "numeric"

nn.data.test.x <- nn.data.test[, -1]
nn.data.test.x <- as.data.frame(lapply(nn.data.test.x, function(x) {
  if (is.factor(x)) as.numeric(as.character(x))
  else if (is.character(x)) as.numeric(x)
  else as.numeric(x)
}))
nn.data.test.x <- as.matrix(nn.data.test.x)
storage.mode(nn.data.test.x) <- "numeric"

nn.data.test.y <- as.numeric(as.character(nn.data.test[, 1]))
nn.data.test.y <- matrix(nn.data.test.y, ncol = 1)
storage.mode(nn.data.test.y) <- "numeric"

cat("Input shape:", ncol(nn.data.train.x), "\n")

# Build model
cat("Building neural network model...\n")
sgd <- optimizer_sgd(learning_rate = 0.02)

nn.model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', 
              input_shape = c(ncol(nn.data.train.x)),
              kernel_regularizer = regularizer_l2(l = 0.005)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 32, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = 0.005)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

nn.model %>% compile(
  optimizer = sgd,
  loss = "mean_squared_error",
  metrics = c("accuracy")
)

cat("Model architecture:\n")
print(nn.model %>% summary())

# Train model
cat("\nTraining neural network...\n")
cat("This may take several minutes...\n")
nn.fit1 <- nn.model %>% fit(
  nn.data.train.x,
  nn.data.train.y,
  epochs = 400,
  batch_size = 512,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate on test set
cat("\nEvaluating on test set...\n")
results <- nn.model %>% evaluate(nn.data.test.x, nn.data.test.y, verbose = 0)

# Get test labels (need to match the split indices)
set.seed(245)
train.indices <- sample(nrow(nn.data), 0.75*nrow(nn.data))
test_labels <- reviews.corpus[-train.indices, ]$LABEL
test_labels <- as.numeric(as.character(test_labels))

# Make predictions
predict.nn <- nn.model %>% predict(nn.data.test.x, verbose = 0)
class.nn <- ifelse(predict.nn > 0.5, "1", "0")
class.nn <- factor(class.nn, levels = c("0", "1"))

# Calculate performance metrics
cat("\n----------------------------------------\n")
cat("Model Performance Metrics\n")
cat("----------------------------------------\n\n")

# Accuracy from evaluation
cat("Test Loss:", round(results$loss, 4), "\n")
cat("Test Accuracy:", round(results$accuracy, 4), "\n")

# Accuracy from predictions
testacc.nn <- mean(class.nn == test_labels)
cat("Prediction Accuracy:", round(testacc.nn, 4), "\n")

# Confusion Matrix
cm <- confusionMatrix(class.nn, factor(test_labels, levels = c("0", "1")))
cat("\nConfusion Matrix:\n")
print(cm$table)

# Detailed metrics
cat("\nDetailed Metrics:\n")
cat("Sensitivity (Recall):", round(cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
cat("Precision:", round(cm$byClass["Precision"], 4), "\n")
cat("F1 Score:", round(cm$byClass["F1"], 4), "\n")

# ROC Curve and AUC
roc_result <- roc(test_labels, as.numeric(predict.nn), quiet = TRUE)
cat("AUC:", round(as.numeric(auc(roc_result)), 4), "\n")

# Visualizations
cat("\nGenerating visualizations...\n")

# 1. Training History Plot
history_df <- data.frame(
  epoch = 1:length(nn.fit1$metrics$loss),
  loss = nn.fit1$metrics$loss,
  accuracy = nn.fit1$metrics$accuracy,
  val_loss = nn.fit1$metrics$val_loss,
  val_accuracy = nn.fit1$metrics$val_accuracy
)

# Loss plot
p1 <- ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Training"), size = 1) +
  geom_line(aes(y = val_loss, color = "Validation"), size = 1) +
  labs(title = "Training History - Loss", x = "Epoch", y = "Loss", color = "Dataset") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "bottom")

# Accuracy plot
p2 <- ggplot(history_df, aes(x = epoch)) +
  geom_line(aes(y = accuracy, color = "Training"), size = 1) +
  geom_line(aes(y = val_accuracy, color = "Validation"), size = 1) +
  labs(title = "Training History - Accuracy", x = "Epoch", y = "Accuracy", color = "Dataset") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        legend.position = "bottom")

# Combine plots
png("figures/model_07_training_history.png", width = 1200, height = 600)
grid.arrange(p1, p2, ncol = 2)
dev.off()

# 2. ROC Curve
png("figures/model_07_roc_curve.png", width = 800, height = 600)
plot(roc_result, main = "ROC Curve - Neural Network (REVIEW_TEXT only)", 
     print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
dev.off()

# 3. Confusion Matrix Heatmap
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted", "Actual", "Freq")
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 8) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Neural Network (REVIEW_TEXT only)",
       x = "Actual", y = "Predicted") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
ggsave("figures/model_07_confusion_matrix.png", width = 6, height = 5, dpi = 300)

# 4. Prediction Probability Distribution
pred_df <- data.frame(
  Probability = as.numeric(predict.nn),
  Label = factor(test_labels, levels = c("0", "1"))
)
ggplot(pred_df, aes(x = Probability, fill = Label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  labs(title = "Prediction Probability Distribution - Neural Network",
       x = "Predicted Probability", y = "Frequency", fill = "Actual Label") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold")) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red")
ggsave("figures/model_07_probability_distribution.png", width = 8, height = 6, dpi = 300)

cat("Visualizations saved to figures/ directory:\n")
cat("  - model_07_training_history.png\n")
cat("  - model_07_roc_curve.png\n")
cat("  - model_07_confusion_matrix.png\n")
cat("  - model_07_probability_distribution.png\n")

# Save model
cat("\nSaving model...\n")
tryCatch({
  # In keras3, use the model's save method with .keras or .h5 extension
  nn.model$save('fakereview.keras')
  cat("Model saved to 'fakereview.keras'\n")
}, error = function(e) {
  cat("Warning: Could not save model. Error:", e$message, "\n")
  cat("Model training completed but could not be saved.\n")
})

cat("\n========================================\n")
cat("Model training and evaluation complete.\n")
cat("========================================\n")

