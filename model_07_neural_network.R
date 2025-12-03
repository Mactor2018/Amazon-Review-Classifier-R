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

# Check initial data types
cat("Initial data types:\n")
print(sapply(nn.data, class))

# One-hot encode product category
cat("One-hot encoding PRODUCT_CATEGORY...\n")
dummy <- acm.disjonctif(nn.data['PRODUCT_CATEGORY'])
nn.data['PRODUCT_CATEGORY'] = NULL
nn.data <- cbind(nn.data, dummy)

# One-hot encode RATING
cat("One-hot encoding RATING...\n")
dummy.rating <- acm.disjonctif(nn.data['RATING'])
nn.data['RATING'] = NULL
nn.data <- cbind(nn.data, dummy.rating)

# Verify all one-hot encoded columns are numeric
cat("Data types after one-hot encoding:\n")
print(sapply(nn.data, class))

# Recode VERIFIED_PURCHASE (convert factor to numeric: N=0, Y=1)
# Handle factor levels that may have been modified by make.names()
# First convert to character, then map to numeric
verif.char <- as.character(nn.data$VERIFIED_PURCHASE)
# Check if levels contain "Y" or "N" (original or modified by make.names)
# make.names() might convert "Y" to "Y" or "Y." depending on context
verif.levels <- unique(verif.char)
# Map to 0/1: first level = 0, second level = 1
# Typically: "N" or "N." -> 0, "Y" or "Y." -> 1
if (length(verif.levels) == 2) {
  # Determine which level corresponds to "Y" (verified purchase)
  # Check if any level contains "Y" (case-insensitive)
  y.level <- verif.levels[grepl("^Y", verif.levels, ignore.case = TRUE) | 
                          grepl("^Y\\.", verif.levels, ignore.case = TRUE)]
  if (length(y.level) > 0) {
    nn.data$VERIFIED_PURCHASE <- ifelse(verif.char == y.level[1], 1, 0)
  } else {
    # Fallback: use first level as 0, second as 1
    nn.data$VERIFIED_PURCHASE <- ifelse(verif.char == verif.levels[1], 0, 1)
  }
} else {
  # Fallback: use as.numeric and subtract 1
  nn.data$VERIFIED_PURCHASE <- as.numeric(nn.data$VERIFIED_PURCHASE) - 1
}
# Ensure it's numeric, not factor
nn.data$VERIFIED_PURCHASE <- as.numeric(nn.data$VERIFIED_PURCHASE)

# Use the same train/test split as defined in data_preprocessing.R
# This ensures all models use the same split for fair comparison
train.indices <- which(rownames(reviews.corpus) %in% rownames(reviews.train))
nn.data.test <- nn.data[-train.indices, ]
nn.data.train <- nn.data[train.indices, ]

# Prepare training and test data
cat("Converting data to numeric matrices...\n")
nn.data.train.x <- nn.data.train[, -1]

# Check data types before conversion
cat("Checking data types before conversion...\n")
data.types <- sapply(nn.data.train.x, class)
factor.cols <- names(data.types)[data.types == "factor"]
if (length(factor.cols) > 0) {
  cat("Warning: Found factor columns:", paste(factor.cols, collapse = ", "), "\n")
  cat("These should have been one-hot encoded. Converting to numeric.\n")
}

# All columns should already be numeric (one-hot encoded or numeric)
# But handle any remaining factors or characters
nn.data.train.x <- as.data.frame(lapply(nn.data.train.x, function(x) {
  if (is.factor(x)) {
    # For factors, convert to numeric properly
    # Try to extract numeric value from factor levels if possible
    char.x <- as.character(x)
    num.x <- suppressWarnings(as.numeric(char.x))
    # If conversion failed, use factor codes
    if (any(is.na(num.x))) {
      num.x <- as.numeric(x)
    }
    num.x
  } else if (is.character(x)) {
    # Try to convert character to numeric, replace NA with 0
    num.x <- suppressWarnings(as.numeric(x))
    num.x[is.na(num.x)] <- 0
    num.x
  } else {
    as.numeric(x)
  }
}))

# Convert to matrix
nn.data.train.x <- as.matrix(nn.data.train.x)
storage.mode(nn.data.train.x) <- "numeric"

# Check for and handle NA/NaN/Inf values
na.count <- sum(is.na(nn.data.train.x))
inf.count <- sum(is.infinite(nn.data.train.x))
if (na.count > 0 || inf.count > 0) {
  cat("Warning: Found", na.count, "NA values and", inf.count, "Inf values in training data. Replacing with 0.\n")
  nn.data.train.x[is.na(nn.data.train.x) | is.infinite(nn.data.train.x)] <- 0
}

nn.data.train.y <- as.numeric(as.character(nn.data.train[, 1]))
nn.data.train.y <- matrix(nn.data.train.y, ncol = 1)
storage.mode(nn.data.train.y) <- "numeric"

nn.data.test.x <- nn.data.test[, -1]
nn.data.test.x <- as.data.frame(lapply(nn.data.test.x, function(x) {
  if (is.factor(x)) {
    char.x <- as.character(x)
    num.x <- suppressWarnings(as.numeric(char.x))
    if (any(is.na(num.x))) {
      num.x <- as.numeric(x)
    }
    num.x
  } else if (is.character(x)) {
    num.x <- suppressWarnings(as.numeric(x))
    num.x[is.na(num.x)] <- 0
    num.x
  } else {
    as.numeric(x)
  }
}))
nn.data.test.x <- as.matrix(nn.data.test.x)
storage.mode(nn.data.test.x) <- "numeric"

# Check for and handle NA/NaN/Inf values in test data
na.count.test <- sum(is.na(nn.data.test.x))
inf.count.test <- sum(is.infinite(nn.data.test.x))
if (na.count.test > 0 || inf.count.test > 0) {
  cat("Warning: Found", na.count.test, "NA values and", inf.count.test, "Inf values in test data. Replacing with 0.\n")
  nn.data.test.x[is.na(nn.data.test.x) | is.infinite(nn.data.test.x)] <- 0
}

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
  loss = "binary_crossentropy",  # Use binary_crossentropy for binary classification
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

