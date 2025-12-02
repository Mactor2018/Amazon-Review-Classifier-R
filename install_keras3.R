# Installation script for keras3 package
# Run this script to install keras3 and its dependencies

cat("========================================\n")
cat("Installing keras3 for Neural Networks\n")
cat("========================================\n\n")

# Step 1: Fix CRAN mirror
cat("Step 1: Setting CRAN mirror...\n")
options(repos = c(CRAN = "https://cloud.r-project.org"))
cat("CRAN mirror set to: https://cloud.r-project.org\n\n")

# Step 2: Install devtools or remotes
cat("Step 2: Installing devtools/remotes...\n")
if (!require("devtools", quietly = TRUE)) {
  cat("Installing devtools...\n")
  install.packages("devtools", repos = "https://cloud.r-project.org")
}

if (!require("remotes", quietly = TRUE)) {
  cat("Installing remotes...\n")
  install.packages("remotes", repos = "https://cloud.r-project.org")
}
cat("devtools/remotes ready.\n\n")

# Step 3: Install keras3 from GitHub
cat("Step 3: Installing keras3 from GitHub...\n")
cat("This may take several minutes...\n")

if (!require("devtools", quietly = TRUE)) {
  if (require("remotes", quietly = TRUE)) {
    cat("Using remotes to install keras3...\n")
    remotes::install_github("rstudio/keras3")
  } else {
    stop("Neither devtools nor remotes is available. Please install one manually.")
  }
} else {
  cat("Using devtools to install keras3...\n")
  devtools::install_github("rstudio/keras3")
}

# Step 4: Verify installation
cat("\nStep 4: Verifying installation...\n")
if (require("keras3", quietly = TRUE)) {
  cat("SUCCESS: keras3 is installed!\n")
  cat("\nNext steps:\n")
  cat("1. Run your neural network model scripts\n")
  cat("2. TensorFlow will be automatically installed on first use\n")
} else {
  cat("WARNING: keras3 installation may have failed.\n")
  cat("Please check the error messages above.\n")
}

cat("\n========================================\n")
cat("Installation script complete.\n")
cat("========================================\n")


