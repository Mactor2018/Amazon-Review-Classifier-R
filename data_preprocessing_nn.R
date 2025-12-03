# Data Preprocessing Script for Neural Network Models
# This script loads and preprocesses the Amazon reviews data
# Used by Neural Network models (model_07, model_08)
# Neural networks require numeric matrices, so categorical variables need one-hot encoding

# Load required libraries
library(tm)
library(magrittr)
library(SnowballC)  # Required for stemDocument function
library(data.table)

# Load data using data.table for robust CSV parsing
cat("Loading data...\n")
reviews <- data.table::fread('data_new.csv', encoding = "UTF-8", quote = "\"")
reviews <- as.data.frame(reviews)

# Data preprocessing
cat("Preprocessing data...\n")
reviews$LABEL <- as.factor(ifelse(reviews$LABEL == '__label1__', 1, 0))
reviews$REVIEW_TITLE <- as.character(reviews$REVIEW_TITLE)
reviews$REVIEW_TEXT <- as.character(reviews$REVIEW_TEXT)

# VERIFIED_PURCHASE: Categorical variable (Y/N), treated as factor
reviews$VERIFIED_PURCHASE <- factor(
  ifelse(reviews$VERIFIED_PURCHASE == 'Y', "Y", "N"),
  levels = c("N", "Y")
)

# RATING: Categorical variable (1-5 stars), treated as factor, not ordinal
reviews$RATING <- factor(reviews$RATING, levels = sort(unique(reviews$RATING)))

# PRODUCT_CATEGORY: Categorical variable, convert to factor
reviews$PRODUCT_CATEGORY <- factor(reviews$PRODUCT_CATEGORY)

# Create corpus and document-term matrix for REVIEW_TEXT
cat("Creating document-term matrix for REVIEW_TEXT...\n")
corpus.text <- VCorpus(VectorSource(reviews$REVIEW_TEXT)) %>% 
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, stopwords()) %>% 
  tm_map(stemDocument)

dtm.text <- DocumentTermMatrix(corpus.text)
dtm.text.cleaned <- removeSparseTerms(dtm.text, .9995)

# Create corpus and document-term matrix for REVIEW_TITLE
cat("Creating document-term matrix for REVIEW_TITLE...\n")
corpus.title <- VCorpus(VectorSource(reviews$REVIEW_TITLE)) %>% 
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, stopwords()) %>% 
  tm_map(stemDocument)

dtm.title <- DocumentTermMatrix(corpus.title)
dtm.title.cleaned <- removeSparseTerms(dtm.title, .9995)

# Convert to matrices
dtm.text.matrix <- as.matrix(dtm.text.cleaned)
dtm.title.matrix <- as.matrix(dtm.title.cleaned)

# Add prefix to distinguish title and text features
colnames(dtm.text.matrix) <- paste0("TEXT_", colnames(dtm.text.matrix))
colnames(dtm.title.matrix) <- paste0("TITLE_", colnames(dtm.title.matrix))

# Select only required columns: LABEL, RATING, VERIFIED_PURCHASE, PRODUCT_CATEGORY
# Keep original columns for reference (will be excluded in model processing)
reviews.selected <- reviews[, c("LABEL", "RATING", "VERIFIED_PURCHASE", "PRODUCT_CATEGORY", 
                                "DOC_ID", "PRODUCT_ID", "PRODUCT_TITLE", "REVIEW_TITLE", "REVIEW_TEXT")]

# Combine with text features
# Column order: LABEL, RATING, VERIFIED_PURCHASE, PRODUCT_CATEGORY, DOC_ID, PRODUCT_ID, 
#               PRODUCT_TITLE, REVIEW_TITLE, REVIEW_TEXT, TEXT_*, TITLE_*
reviews.corpus <- data.frame(reviews.selected, dtm.text.matrix, dtm.title.matrix)

# Split data (same split as data_preprocessing.R for consistency across all models)
set.seed(245)
train <- sample(nrow(reviews.corpus), .75 * nrow(reviews.corpus))
reviews.train <- reviews.corpus[train,]
reviews.test <- reviews.corpus[-train,]

cat("Data preprocessing complete.\n")
cat("Training set size:", nrow(reviews.train), "\n")
cat("Test set size:", nrow(reviews.test), "\n")
cat("Total features:", ncol(reviews.corpus), "\n")
cat("Column order: LABEL (1), RATING (2), VERIFIED_PURCHASE (3), PRODUCT_CATEGORY (4),")
cat(" DOC_ID (5), PRODUCT_ID (6), PRODUCT_TITLE (7), REVIEW_TITLE (8), REVIEW_TEXT (9),")
cat(" then TEXT_* and TITLE_* features\n")

