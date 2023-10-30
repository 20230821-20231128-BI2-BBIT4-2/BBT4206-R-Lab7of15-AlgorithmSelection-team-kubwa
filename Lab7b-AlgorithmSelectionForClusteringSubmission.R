# *****************************************************************************
# Lab 7.b.: Algorithm Selection for Clustering ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# Clustering is a type of unsupervised machine learning technique that aims to
# group similar data points together into clusters or segments based on certain
# characteristics or similarities, without the need for predefined labels or
# target outcomes. In clustering, the goal is to discover hidden patterns or
# structures in data and to create natural groupings of data points.

## Some Applications of Clustering ----

### 1. Customer segmentation ----
# Grouping customers into segments with similar purchase behavior or
# demographics.

### 2. Anomaly detection ----
# Identifying unusual or outlier data points.

### 3. Document categorization ----
# Clustering documents based on their content to
# discover topics or themes.

### 4. Image segmentation ----
# Segmenting an image into different regions based on
# color or texture.

### 5. Genetic analysis ----
# Clustering genes or proteins with similar functions.

# In R, there are several clustering algorithms you can use. The choice of
# clustering algorithm depends on the nature of your data, the number of
# clusters you want to find, and the specific requirements of your analysis.

## Popular clustering algorithms ----

### 1. K-Means Clustering ----
# K-means is a partitioning-based clustering algorithm that divides the data
# into K non-overlapping clusters. It aims to minimize the sum of squared
# distances from data points to the cluster center.
# Example usage: kmeans_result <- kmeans(data, centers = K)

### 2. Hierarchical Clustering ----
# Hierarchical clustering creates a hierarchy of clusters by repeatedly merging
# or splitting existing clusters. It does not require specifying the number of
# clusters in advance.
# Example usage: hclust_result <- hclust(dist(data))

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) ----
# DBSCAN groups data points based on their density. It can discover clusters of
# arbitrary shapes and identify outliers as noise.
# Example usage: dbscan_result <- dbscan(data, eps = 0.5, minPts = 5)

### 4. Agglomerative Clustering ----
# Agglomerative clustering is a hierarchical clustering algorithm that starts
# with individual data points as clusters and merges them gradually.
# Example usage: agnes_result <- agnes(data)

### 5. OPTICS (Ordering Points To Identify the Clustering Structure) ----
# OPTICS is a density-based clustering algorithm that provides a visualization
# of the cluster structure of the data.

### 6. Gaussian Mixture Models (GMM) ----
# GMM is a probabilistic model that assumes that the data is generated from a
# mixture of several Gaussian distributions.
# Example usage: gmm_result <- Mclust(data)

# The choice of clustering algorithm depends on factors such as data
# distribution, the number of clusters, noise tolerance, and the problem you
# are trying to solve. It is often a good practice to try multiple algorithms
# and assess their results to determine the best approach for your specific
# data and objectives.

## Performance Metrics for Clustering ----
# Given that clustering is unsupervised learning and it does not use labeled
# data, we cannot calculate performance metrics like accuracy, Cohen's Kappa,
# AUC, LogLoss, RMSE, R squared, etc., to compare different algorithms and
# their models. As a result, assessing the performance of clustering models
# is challenging and subjective.

# The subjective assessment involves determining whether the clustering model
# is interpretable, whether the output of the clustering model is useful, and
# whether the clustering model has led to the discovery of new patterns in the
# data.

## The K-Means Clustering Algorithm ----
# The K-Means clustering algorithm is a popular algorithm for clustering tasks
# because of its intuition and ease of implementation.

# K-Means is a centroid-based algorithm where the ML Engineer must define the
# required number of clusters to be created. The number of clusters can be
# informed by the business use-case or through trial and error.

### Steps in K-Means Clustering ----
#     1. Choose the number of clusters, k.
#     2. Select k points (clusters of size 1) at random. These are referred to
#        as the centroids.
#     3. Calculate the distance between each point and the centroid and assign
#        each data point to the closest centroid.
#     4. Calculate the centroid (mean position) for each cluster based on the
#        assigned data points. This will change the position of the centroid.
#     5. Repeat steps 3–4 until the clusters do not change or until the
#        maximum number of iterations is reached.

# Watch the following video: https://youtu.be/4b5d3muPQmA?si=7F9d2A6_MlqSsag2

# STEP 1. Install and Load the Required Packages ----
## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## corrplot ----
if (require("corrplot")) {
  require("corrplot")
} else {
  install.packages("corrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggcorrplot ----
if (require("ggcorrplot")) {
  require("ggcorrplot")
} else {
  install.packages("ggcorrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load the Dataset ----
# Source: http://insideairbnb.com/cape-town/
# Save the dataset as "cubic.csv" inside the data folder

# License: https://creativecommons.org/licenses/by/4.0/

library(readr)
cubic <- read_csv("data/cubic.csv")
View(cubic)

cubic$cut <- factor(cubic$cut)

str(cubic)
dim(cubic)
head(cubic)
summary(cubic)

# STEP 3. Check for Missing Data and Address it ----
# Are there missing values in the dataset?
any_na(cubic)

# How many?
n_miss(cubic)

# What is the proportion of missing data in the entire dataset?
prop_miss(cubic)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(cubic)

# Which variables contain the most missing values?
gg_miss_var(cubic)


# Where are missing values located (the shaded regions in the plot)?
vis_miss(cubic) +
  theme(axis.text.x = element_text(angle = 80))

## OPTION 1: Remove the observations with missing values ----
# We can decide to remove all the observations that have missing values
# as follows:
cubic_removed_obs <- cubic %>% filter(complete.cases(.))

# The initial dataset had 21,120 observations and 16 variables
dim(cubic)

# The filtered dataset has 16,205 observations and 16 variables
dim(cubic_removed_obs)

# Are there missing values in the dataset?
any_na(cubic_removed_obs)

## OPTION 2: Remove the variables with missing values ----
# Alternatively, we can decide to remove the 2 variables that have missing data
cubic_removed_obs <-
  cubic %>%
  dplyr::filter(complete.cases(.))

# The initial dataset had 21,120 observations and 16 variables
dim(cubic)

# The filtered dataset has 21,120 observations and 14 variables
dim(cubic_removed_obs)

# Are there missing values in the dataset?
any_na(cubic_removed_obs)

## OPTION 3: Perform Data Imputation ----

# CAUTION:
# 1. Avoid Over-imputation:
# Be cautious when imputing dates, especially if it is
# Missing Not at Random (MNAR).
# Over-Imputing can introduce bias into your analysis. For example, if dates
# are missing because of a specific event or condition, imputing dates might
# not accurately represent the data.

# 2. Consider the Business Context:
# Dates often have a significant business or domain context. Imputing dates
# may not always be appropriate, as it might distort the interpretation of
# your data. For example, imputing order dates could lead to incorrect insights
# into seasonality trends.

# library(mice) # nolint
# somewhat_correlated_variables <- quickpred(airbnb_cape_town, mincor = 0.3) # nolint

# airbnb_cape_town_imputed <-
#   mice(airbnb_cape_town, m = 11, method = "pmm",
#        seed = 7, # nolint
#        predictorMatrix = somewhat_correlated_variables)

# The choice left is between OPTION 1 and OPTION 2:
# Considering that the 2 variables had 23.3% missing data each,
# we decide to remove the observations that have the missing data (OPTION 1)
# as opposed to removing the entire variable just because 23.3% of its values
# are missing (OPTION 2).

# STEP 4. Perform EDA and Feature Selection ----
## Compute the correlations between variables ----
# We identify the correlated variables because it is these correlated variables
# that can then be used to identify the clusters.

# Create a correlation matrix
# Option 1: Basic Table
cor(cubic_removed_obs[, c(1, 2, 7, 8, 9, 10, 11)]) %>%
  View()

# Option 2: Basic Plot
cor(cubic_removed_obs[, c(1, 2, 7, 8, 9, 10, 11)]) %>%
  corrplot(method = "square")

# Option 3: Fancy Plot using ggplot2
corr_matrix <- cor(cubic_removed_obs[, c(1, 2, 7, 8, 9, 10, 11)])

p <- ggplot2::ggplot(data = reshape2::melt(corr_matrix),
                     ggplot2::aes(Var1, Var2, fill = value)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = label_wrap(label, width = 10)),
                     size = 4) +
  ggplot2::theme_minimal() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

# The correlation plot shows a -0.06 correlation between the price and the
# reviews_per_month. This is worth investigating further if the intention
# of the business is to create clusters based on price.

# Room_type, neighbourhood, date and other non-numeric variables and
# categorical variables are not included in the correlation, but they can be
# used as an additional dimension when plotting the scatter plot during EDA.

## Plot the scatter plots ----
# A scatter plot to show the table against price
# based on quality
ggplot(cubic_removed_obs,
       aes(table, price,
           color = cut,
           shape = cut)) +
  geom_point(alpha = 0.5) +
  xlab("Table") +
  ylab("Daily Price")

# A scatter plot to show table against price
# based on clarity
ggplot(cubic_removed_obs,
       aes(table, price,
           color = clarity)) +
  geom_point(alpha = 0.5) +
  xlab("Table") +
  ylab("Daily Price")

# A scatter plot to show the carat against price
# based on quality
ggplot(cubic_removed_obs,
       aes(carat, price,
           color = cut,
           shape = cut)) +
  geom_point(alpha = 0.5) +
  xlab("Carat") +
  ylab("Daily Price")

# A scatter plot to show the variable y against carat based on quality
ggplot(cubic_removed_obs,
       aes(y, carat,
           color = cut,
           shape = cut)) +
  geom_point(alpha = 0.5) +
  xlab("y") +
  ylab("carat")


## Transform the data ----
# The K Means Clustering algorithm performs better when data transformation has
# been applied. This helps to standardize the data making it easier to compare
# multiple variables.



summary(cubic_removed_obs)
model_of_the_transform <- preProcess(cubic_removed_obs,
                                     method = c("scale", "center"))
print(model_of_the_transform)
cubic_removed_obs_std <- predict(model_of_the_transform, # nolint
                                 cubic_removed_obs)
summary(cubic_removed_obs_std)
sapply(cubic_removed_obs_std[, c(1, 2, 7, 8, 9, 10, 11)], sd)

## Select the features to use to create the clusters ----
# OPTION 1: Use all the numeric variables to create the clusters
cubic_vars <-
  cubic_removed_obs_std[, c(1, 2, 7, 8, 9, 10, 11)]

# OPTION 2: Use only the most significant variables to create the clusters
# This can be informed by feature selection, or by the business case.

# Suppose that the business case is that we need to know the clusters that
# are related to the number of listings a host owns against the listings'
# popularity (measured by number of reviews).

# We need to find the ideal number of listings to own without negatively
# impacting the popularity of the listing.

cubic_vars <-
  cubic_removed_obs_std[, c("carat",
                                       "x")]

# STEP 5. Create the clusters using the K-Means Clustering Algorithm ----
# We start with a random guess of the number of clusters we need

set.seed(7)
kmeans_cluster <- kmeans(cubic_vars, centers = 3, nstart = 20)

# We then decide the maximum number of clusters to investigate
n_clusters <- 8

# Initialize total within sum of squares error: wss
wss <- numeric(n_clusters)

set.seed(7)

# Investigate 1 to n possible clusters (where n is the maximum number of 
# clusters that we want to investigate)
for (i in 1:n_clusters) {
  # Use the K Means cluster algorithm to create each cluster
  kmeans_cluster <- kmeans(cubic_vars, centers = i, nstart = 20)
  # Save the within cluster sum of squares
  wss[i] <- kmeans_cluster$tot.withinss
}

## Plot a scree plot ----
# The scree plot should help you to note when additional clusters do not make
# any significant difference (the plateau).
# Load the entire tidyverse, which includes dplyr, tidyr, and tibble
# Install and load the dplyr package
install.packages("dplyr")
library(dplyr)

# Install and load the tidyr package
install.packages("tidyr")
library(tidyr)



# You can install and load other packages from the 'tidyverse' in a similar manner.


wss_df <- tibble(clusters = 1:n_clusters, wss = wss)

scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
  geom_point(size = 4) +
  geom_line() +
  scale_x_continuous(breaks = c(2, 4, 6, 8)) +
  xlab("Number of Clusters")

scree_plot

# We can add guides to make it easier to identify the plateau (or "elbow").
scree_plot +
  geom_hline(
    yintercept = wss,
    linetype = "dashed",
    col = c(rep("#000000", 5), "#FF0000", rep("#000000", 2))
  )

# The plateau is reached at 6 clusters.
# We therefore create the final cluster with 6 clusters
# (not the initial 3 used at the beginning of this STEP.)
k <- 6
set.seed(7)
# Build model with k clusters: kmeans_cluster
kmeans_cluster <- kmeans(cubic_vars, centers = k, nstart = 20)

# STEP 6. Add the cluster number as a label for each observation ----
cubic_removed_obs$cluster_id <- factor(kmeans_cluster$cluster)

## View the results by plotting scatter plots with the labelled cluster ----
ggplot(cubic_removed_obs, aes(carat, price,
                                         color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("carat") +
  ylab("Price")

ggplot(cubic_removed_obs,
       aes(carat, table, color = ...1)) +
  geom_point(alpha = 0.5) +
  xlab("carat") +
  ylab("table")

ggplot(cubic_removed_obs,
       aes(carat, table,
           color = ...1)) +
  geom_point(alpha = 0.5) +
  xlab("carat") +
  ylab("table")

# Note on Clustering for both Descriptive and Predictive Data Analytics ----
# Clustering can be used for both descriptive and predictive analytics.
# It is more commonly used around Exploratory Data Analysis which is
# descriptive analytics.

# The results of clustering, i.e., a label of the cluster can be fed as input
# to a supervised learning algorithm. The trained model can then be used to
# predict the cluster that a new observation will belong to.

# References ----
## Ali, M. (2022, August). Clustering in Machine Learning: 5 Essential Clustering Algorithms. https://www.datacamp.com/blog/clustering-in-machine-learning-5-essential-clustering-algorithms # nolint ----

## Cox, M., Morris, J., & Higgins, T. (2023). Airbnb Dataset for Cape Town in South Africa [Dataset; CSV]. Inside Airbnb. http://insideairbnb.com/cape-town/ # nolint ----