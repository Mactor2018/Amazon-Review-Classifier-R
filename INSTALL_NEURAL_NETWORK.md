# Installing Keras3 for Neural Network Models

The neural network models (model_07 and model_08) require the `keras3` package. Follow these steps to install it:

## Step 1: Fix CRAN Mirror Issue

If you're getting "unable to access index for repository" errors, first fix your CRAN mirror:

```r
# Set a working CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Or choose a mirror interactively
chooseCRANmirror()
```

## Step 2: Install Required Dependencies

```r
# Install devtools or remotes (try both if one fails)
install.packages("devtools")
# OR
install.packages("remotes")
```

## Step 3: Install keras3 Package

### Option A: Install from GitHub (Recommended for R 4.4)
```r
# Using devtools
devtools::install_github("rstudio/keras3")

# OR using remotes
remotes::install_github("rstudio/keras3")
```

### Option B: Install from CRAN (if available for your R version)
```r
install.packages("keras3")
```

**Note**: If you're using R 4.4, keras3 might not be available on CRAN yet. Use Option A (GitHub) instead.

## Quick Installation Script

For easiest installation, run the provided script:

```bash
Rscript install_keras3.R
```

Or in R:
```r
source("install_keras3.R")
```

## Step 2: Install TensorFlow Backend

After installing keras3, TensorFlow will be automatically installed when you first run:
```r
library(keras3)
use_backend("tensorflow")
```

This happens automatically when you run the neural network model scripts.

## Step 3: Verify Installation

You can verify the installation by running:
```r
library(keras3)
use_backend("tensorflow")
cat("Keras3 and TensorFlow are ready!\n")
```

## Troubleshooting

### CRAN Mirror Issues:
If you see "unable to access index for repository" errors:
```r
# Set a working CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
# Then try installing again
```

### Package Not Available for R Version:
If you see "package 'keras3' is not available for this version of R":
- keras3 might require R 4.5+ or need to be installed from GitHub
- Use the GitHub installation method: `devtools::install_github("rstudio/keras3")`

### If devtools/remotes installation fails:
1. Try installing from a different CRAN mirror:
   ```r
   options(repos = c(CRAN = "https://cloud.r-project.org"))
   install.packages("devtools")
   ```

2. Or download and install manually from:
   - devtools: https://cran.r-project.org/package=devtools
   - remotes: https://cran.r-project.org/package=remotes

### If TensorFlow installation fails:
1. Make sure you have Python installed on your system
2. Install reticulate package: `install.packages("reticulate")`
3. Try installing TensorFlow manually:
   ```r
   library(reticulate)
   py_install("tensorflow")
   ```

### If you get "no package called 'keras3'" error:
- Make sure you've installed keras3 from GitHub successfully
- Check if the package is loaded: `library(keras3)`
- Try restarting R session and installing again
- Verify installation: `packageVersion("keras3")`

## Alternative: Skip Neural Network Models

If you don't need to run the neural network models, you can skip them and run only the other models:
- model_01_glm_lasso_min.R
- model_02_glm_lasso_1se.R
- model_03_random_forest.R
- model_04_ranger.R
- model_05_xgboost.R
- model_06_simple_glm.R

These models don't require keras3.

