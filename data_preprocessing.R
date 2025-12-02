# Data Preprocessing Script
# This script loads and preprocesses the Amazon reviews data
# Used by all model scripts

# Load required libraries
library(tm)
library(magrittr)
library(SnowballC)  # Required for stemDocument function
library(ggplot2)

# Load data
cat("Loading data...\n")
reviews <- read.csv('data_new.csv')

# Data preprocessing
cat("Preprocessing data...\n")
reviews$LABEL <- as.factor(ifelse(reviews$LABEL == '__label1__', 1, 0))
reviews$REVIEW_TITLE <- as.character(reviews$REVIEW_TITLE)
reviews$REVIEW_TEXT <- as.character(reviews$REVIEW_TEXT)
reviews$VERIFIED_PURCHASE <- as.numeric(ifelse(reviews$VERIFIED_PURCHASE == 'Y', 1, 0))
reviews$RATING <- factor(reviews$RATING, levels = sort(unique(reviews$RATING)))

# Create corpus and document-term matrix
cat("Creating document-term matrix...\n")
corpus <- VCorpus(VectorSource(reviews$REVIEW_TEXT)) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(stemDocument)

dtm <- DocumentTermMatrix(corpus)
dtm.cleaned <- removeSparseTerms(dtm, .9995)

# Combine with original data
reviews.corpus <- data.frame(reviews, as.matrix(dtm.cleaned))

# Visualize 
# Create figures directory if it doesn't exist
if (!dir.exists("figures")) {
  dir.create("figures")
}

cat("Generating data visualization plots...\n")

# 1. Distribution plots with improved styling
# a. LABEL distribution
ggplot(reviews.corpus, aes(x = LABEL, fill = LABEL)) +
  geom_bar(alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#E74C3C", "1" = "#3498DB"), guide = "none") +
  labs(title = "Distribution of Labels",
       x = "Label", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12)) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, size = 4)
ggsave("figures/data_preprocessing_label_distribution.png", width = 8, height = 6, dpi = 300)

# b. RATING distribution
ggplot(reviews.corpus, aes(x = RATING, fill = RATING)) +
  geom_bar(alpha = 0.8) +
  scale_fill_brewer(palette = "RdYlGn", guide = "none") +
  labs(title = "Distribution of Ratings",
       x = "Rating", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12)) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, size = 4)
ggsave("figures/data_preprocessing_rating_distribution.png", width = 8, height = 6, dpi = 300)

# c. VERIFIED_PURCHASE distribution
reviews.corpus$VERIFIED_PURCHASE_FACTOR <- factor(reviews.corpus$VERIFIED_PURCHASE, 
                                                    levels = c(0, 1),
                                                    labels = c("No", "Yes"))
ggplot(reviews.corpus, aes(x = VERIFIED_PURCHASE_FACTOR, fill = VERIFIED_PURCHASE_FACTOR)) +
  geom_bar(alpha = 0.8) +
  scale_fill_manual(values = c("No" = "#95A5A6", "Yes" = "#27AE60"), guide = "none") +
  labs(title = "Distribution of Verified Purchases",
       x = "Verified Purchase", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12)) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, size = 4)
ggsave("figures/data_preprocessing_verified_purchases_distribution.png", width = 8, height = 6, dpi = 300)

# 2. Relationship visualizations using appropriate chart types
# a. LABEL vs RATING: Stacked bar chart showing proportions
label_rating_data <- as.data.frame(table(reviews.corpus$LABEL, reviews.corpus$RATING))
names(label_rating_data) <- c("Label", "Rating", "Count")
label_rating_data$Label <- factor(label_rating_data$Label, levels = c("0", "1"), labels = c("Fake (0)", "Real (1)"))

# Calculate percentages for each label
label_totals <- aggregate(Count ~ Label, label_rating_data, sum)
label_rating_data$Percentage <- apply(label_rating_data, 1, function(x) {
  total <- label_totals[label_totals$Label == x["Label"], "Count"]
  round(as.numeric(x["Count"]) / total * 100, 1)
})

ggplot(label_rating_data, aes(x = Rating, y = Count, fill = Label)) +
  geom_bar(stat = "identity", position = "stack", alpha = 0.8) +
  scale_fill_manual(values = c("Fake (0)" = "#E74C3C", "Real (1)" = "#3498DB"),
                    name = "Label") +
  labs(title = "Label Distribution by Rating",
       x = "Rating", y = "Count",
       subtitle = "Stacked bar chart showing label distribution across different ratings") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray50"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        legend.position = "right") +
  geom_text(aes(label = paste0(Count, "\n(", Percentage, "%)")), 
            position = position_stack(vjust = 0.5), size = 3, color = "white", fontface = "bold")
ggsave("figures/data_preprocessing_label_rating_correlation.png", width = 10, height = 6, dpi = 300)

# b. LABEL vs RATING: Grouped bar chart for better comparison
ggplot(label_rating_data, aes(x = Rating, y = Count, fill = Label)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Fake (0)" = "#E74C3C", "Real (1)" = "#3498DB"),
                    name = "Label") +
  labs(title = "Label Count by Rating",
       x = "Rating", y = "Count",
       subtitle = "Grouped bar chart comparing label counts across ratings") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray50"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        legend.position = "right") +
  geom_text(aes(label = Count), position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3.5, fontface = "bold")
ggsave("figures/data_preprocessing_label_rating_grouped.png", width = 10, height = 6, dpi = 300)

# c. LABEL vs VERIFIED_PURCHASE: Stacked bar chart
label_verified_data <- as.data.frame(table(reviews.corpus$LABEL, reviews.corpus$VERIFIED_PURCHASE_FACTOR))
names(label_verified_data) <- c("Label", "VerifiedPurchase", "Count")
label_verified_data$Label <- factor(label_verified_data$Label, levels = c("0", "1"), labels = c("Fake (0)", "Real (1)"))

# Calculate percentages
verified_totals <- aggregate(Count ~ Label, label_verified_data, sum)
label_verified_data$Percentage <- apply(label_verified_data, 1, function(x) {
  total <- verified_totals[verified_totals$Label == x["Label"], "Count"]
  round(as.numeric(x["Count"]) / total * 100, 1)
})

ggplot(label_verified_data, aes(x = VerifiedPurchase, y = Count, fill = Label)) +
  geom_bar(stat = "identity", position = "stack", alpha = 0.8) +
  scale_fill_manual(values = c("Fake (0)" = "#E74C3C", "Real (1)" = "#3498DB"),
                    name = "Label") +
  labs(title = "Label Distribution by Verified Purchase Status",
       x = "Verified Purchase", y = "Count",
       subtitle = "Stacked bar chart showing label distribution across verified purchase status") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray50"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        legend.position = "right") +
  geom_text(aes(label = paste0(Count, "\n(", Percentage, "%)")), 
            position = position_stack(vjust = 0.5), size = 4, color = "white", fontface = "bold")
ggsave("figures/data_preprocessing_label_verified_purchases_correlation.png", width = 8, height = 6, dpi = 300)

# d. LABEL vs VERIFIED_PURCHASE: Grouped bar chart
ggplot(label_verified_data, aes(x = VerifiedPurchase, y = Count, fill = Label)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Fake (0)" = "#E74C3C", "Real (1)" = "#3498DB"),
                    name = "Label") +
  labs(title = "Label Count by Verified Purchase Status",
       x = "Verified Purchase", y = "Count",
       subtitle = "Grouped bar chart comparing label counts across verified purchase status") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 10, color = "gray50"),
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        legend.position = "right") +
  geom_text(aes(label = Count), position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 4, fontface = "bold")
ggsave("figures/data_preprocessing_label_verified_purchases_grouped.png", width = 8, height = 6, dpi = 300)

cat("Data visualization plots saved to figures/ directory.\n")


# Split data
set.seed(245)
train <- sample(nrow(reviews.corpus), .75 * nrow(reviews.corpus))
reviews.train <- reviews.corpus[train,]
reviews.test <- reviews.corpus[-train,]

cat("Data preprocessing complete.\n")
cat("Training set size:", nrow(reviews.train), "\n")
cat("Test set size:", nrow(reviews.test), "\n")
cat("Total features:", ncol(reviews.corpus), "\n")

