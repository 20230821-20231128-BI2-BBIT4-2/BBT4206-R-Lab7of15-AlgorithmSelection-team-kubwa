Business Intelligence Lab Submission Markdown
================
<team kubwa>
\<23/10/2023\>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Loading the Loan Status Train Imputed
  Dataset](#loading-the-loan-status-train-imputed-dataset)
  - [Description of the Dataset](#description-of-the-dataset)
- [\<Checking for missing values\>](#checking-for-missing-values)
  - [\<Determine the Baseline
    Accuracy\>](#determine-the-baseline-accuracy)
  - [\<Split the dataset\>](#split-the-dataset)
  - [\<RMSE, R Squared, and MAE\>](#rmse-r-squared-and-mae)
  - [\<Area Under ROC Curve\>](#area-under-roc-curve)
  - [\<Logarithmic Loss \>](#logarithmic-loss-)

# Student Details

|                                                   |                                                                                                                                                                                                                                                                                                                                                                |     |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** | *\<list one Student name, class group (just the letter; A, B, or C), and ID per line, e.g., 123456 - A - John Leposo; you should be between 2 and 5 members per group\>* \| \| 1. 128998 - B - Crispus Nzano \| \| 2. 134100 - B - Timothy Obosi \| \| 3. 134092 - B - Alison Kuria \| 4. 135269 - B - Clifford Kipchumba \| \| 5. 112826 - B - Matu Ngatia \| |     |
| **GitHub Classroom Group Name**                   | Team Kubwa \|                                                                                                                                                                                                                                                                                                                                                  |     |
| **Course Code**                                   | BBT4206                                                                                                                                                                                                                                                                                                                                                        |     |
| **Course Name**                                   | Business Intelligence II                                                                                                                                                                                                                                                                                                                                       |     |
| **Program**                                       | Bachelor of Business Information Technology                                                                                                                                                                                                                                                                                                                    |     |
| **Semester Duration**                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023                                                                                                                                                                                                                                                                                                   |     |

# Setup Chunk

We start by installing all the required packages

``` r
## formatR - Required to format R code in the markdown ----

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# Resampling methods are techniques that can be used to improve the performance
# and reliability of machine learning algorithms. They work by creating
# multiple training sets from the original training set. The model is then
# trained on each training set, and the results are averaged. This helps to
# reduce overfitting and improve the model's generalization performance.

# Resampling methods include:
## Splitting the dataset into train and test sets ----
## Bootstrapping (sampling with replacement) ----
## Basic k-fold cross validation ----
## Repeated cross validation ----
## Leave One Out Cross-Validation (LOOCV) ----

# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (!is.element("NHANES", installed.packages()[, 1])) {
  install.packages("NHANES", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("NHANES")

## dplyr ----
if (!is.element("dplyr", installed.packages()[, 1])) {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("dplyr")

## naniar ----
# Documentation:
#   https://cran.r-project.org/package=naniar or
#   https://www.rdocumentation.org/packages/naniar/versions/1.0.0
if (!is.element("naniar", installed.packages()[, 1])) {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("naniar")

## ggplot2 ----
# We require the "ggplot2" package to create more appealing visualizations
if (!is.element("ggplot2", installed.packages()[, 1])) {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("ggplot2")

## MICE ----
# We use the MICE package to perform data imputation
if (!is.element("mice", installed.packages()[, 1])) {
  install.packages("mice", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("mice")

## Amelia ----
if (!is.element("Amelia", installed.packages()[, 1])) {
  install.packages("Amelia", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
require("Amelia")
```

------------------------------------------------------------------------

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

------------------------------------------------------------------------

**Note:** the following “*R Markdown*” options have been set as the
defaults in this markdown:

> output:  
>   
> github_document:  
> toc: yes  
> toc_depth: 4  
> fig_width: 6  
> fig_height: 4  
> df_print: default  
>   
> editor_options:  
> chunk_output_type: console

# Loading the Loan Status Train Imputed Dataset

The Dataset is then loaded.

``` r
library(readr)
train_imputed <- read_csv("C:/Users/Cris/github-classroom/BBT4206-R-Lab6of15-EvaluationMetrics-team-kubwa/data/train_imputed.csv")
```

    ## Rows: 614 Columns: 12
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (7): Gender, Married, Dependents, Education, SelfEmployed, PropertyArea,...
    ## dbl (5): ApplicantIncome, CoapplicantIncome, LoanAmount, LoanAmountTerm, Cre...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(train_imputed)
```

## Description of the Dataset

We then display the number of observations and number of variables. 12
Variables and 614 observations.

``` r
dim(train_imputed)
```

    ## [1] 614  12

Next, we display the quartiles for each numeric
variable<span id="highlight" style="color: blue">*… this is the process
of **“storytelling using the data.”** The goal is to analyse the
PimaIndians dataset and try to train a model to make predictions( which
model is most suited for this dataset).*</span>

``` r
summary(train_imputed)
```

    ##     Gender            Married           Dependents         Education        
    ##  Length:614         Length:614         Length:614         Length:614        
    ##  Class :character   Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character   Mode  :character  
    ##                                                                             
    ##                                                                             
    ##                                                                             
    ##  SelfEmployed       ApplicantIncome CoapplicantIncome   LoanAmount   
    ##  Length:614         Min.   :  150   Min.   :    0     Min.   :  150  
    ##  Class :character   1st Qu.: 2878   1st Qu.:    0     1st Qu.: 2875  
    ##  Mode  :character   Median : 3812   Median : 1188     Median : 3768  
    ##                     Mean   : 5403   Mean   : 1621     Mean   : 5371  
    ##                     3rd Qu.: 5795   3rd Qu.: 2297     3rd Qu.: 5746  
    ##                     Max.   :81000   Max.   :41667     Max.   :81000  
    ##  LoanAmountTerm  CreditHistory   PropertyArea          Status         
    ##  Min.   : 12.0   Min.   :0.000   Length:614         Length:614        
    ##  1st Qu.:360.0   1st Qu.:1.000   Class :character   Class :character  
    ##  Median :360.0   Median :1.000   Mode  :character   Mode  :character  
    ##  Mean   :342.3   Mean   :0.855                                        
    ##  3rd Qu.:360.0   3rd Qu.:1.000                                        
    ##  Max.   :480.0   Max.   :1.000

# \<Checking for missing values\>

We first check if the dataset has missing values and it was found that
there were therefore imputation was erformed on the dataset. There
should be no missing values.

``` r
# Are there missing values in the dataset?
any_na(train_imputed)
```

    ## [1] FALSE

``` r
# How many?
n_miss(train_imputed)
```

    ## [1] 0

``` r
# What is the percentage of missing data in the entire dataset?
prop_miss(train_imputed)
```

    ## [1] 0

``` r
# How many missing values does each variable have?
train_imputed %>%
    is.na() %>%
    colSums()
```

    ##            Gender           Married        Dependents         Education 
    ##                 0                 0                 0                 0 
    ##      SelfEmployed   ApplicantIncome CoapplicantIncome        LoanAmount 
    ##                 0                 0                 0                 0 
    ##    LoanAmountTerm     CreditHistory      PropertyArea            Status 
    ##                 0                 0                 0                 0

``` r
# What is the number and percentage of missing values grouped by each variable?
miss_var_summary(train_imputed)
```

    ## # A tibble: 12 × 3
    ##    variable          n_miss pct_miss
    ##    <chr>              <int>    <dbl>
    ##  1 Gender                 0        0
    ##  2 Married                0        0
    ##  3 Dependents             0        0
    ##  4 Education              0        0
    ##  5 SelfEmployed           0        0
    ##  6 ApplicantIncome        0        0
    ##  7 CoapplicantIncome      0        0
    ##  8 LoanAmount             0        0
    ##  9 LoanAmountTerm         0        0
    ## 10 CreditHistory          0        0
    ## 11 PropertyArea           0        0
    ## 12 Status                 0        0

``` r
# What is the number and percentage of missing values grouped by each
# observation?
miss_case_summary(train_imputed)
```

    ## # A tibble: 614 × 3
    ##     case n_miss pct_miss
    ##    <int>  <int>    <dbl>
    ##  1     1      0        0
    ##  2     2      0        0
    ##  3     3      0        0
    ##  4     4      0        0
    ##  5     5      0        0
    ##  6     6      0        0
    ##  7     7      0        0
    ##  8     8      0        0
    ##  9     9      0        0
    ## 10    10      0        0
    ## # ℹ 604 more rows

``` r
# Which variables contain the most missing values?
gg_miss_var(train_imputed)
```

![](Lab7a-AlgorithmSelectionForClassificationAndRegressionSubmission_files/figure-gfm/Your%20Sixth%20Code%20Chunk-1.png)<!-- -->

``` r
# Where are missing values located (the shaded regions in the plot)?
vis_miss(train_imputed) + theme(axis.text.x = element_text(angle = 80))
```

![](Lab7a-AlgorithmSelectionForClassificationAndRegressionSubmission_files/figure-gfm/Your%20Sixth%20Code%20Chunk-2.png)<!-- -->

## \<Determine the Baseline Accuracy\>

Identify the number of instances that belong to each class (distribution
or class breakdown).

``` r
train_Status_freq <- train_imputed$Status
cbind(frequency = table(train_Status_freq), percentage = prop.table(table(train_Status_freq)) *
    100)
```

    ##   frequency percentage
    ## N       192   31.27036
    ## Y       422   68.72964

## \<Split the dataset\>

We define a 75:25 train:test data split of the dataset. That is, 75% of
the original data will be used to train the model and 25% of the
original data will be used to test the model.

``` r
train_index <- createDataPartition(train_imputed$Status, p = 0.75, list = FALSE)
status_train <- train_imputed[train_index, ]
status_test <- train_imputed[-train_index, ]

## 1.d. Train the Model ---- We apply the 5-fold cross validation resampling
## method
train_control <- trainControl(method = "cv", number = 5)

# We then train a Generalized Linear Model to predict the value of medv.

# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same 'random' numbers.
set.seed(7)
status_model_glm <- train(Status ~ ., data = status_train, method = "glm", metric = "Accuracy",
    trControl = train_control)

## 1.e. Display the Model's Performance ---- Option 1: Use the metric
## calculated by caret when training the model ---- The results show an
## accuracy of approximately 77% (slightly above the baseline accuracy) and a
## Kappa of approximately 49%.
print(status_model_glm)
```

    ## Generalized Linear Model 
    ## 
    ## 461 samples
    ##  11 predictor
    ##   2 classes: 'N', 'Y' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 370, 369, 369, 368, 368 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.8091544  0.4776499

``` r
### Option 2: Compute the metric yourself using the test dataset ---- A
### confusion matrix is useful for multi-class classification problems.  Please
### watch the following video first: https://youtu.be/Kdsp6soqA7o

# The Confusion Matrix is a type of matrix which is used to visualize the
# predicted values against the actual Values. The row headers in the confusion
# matrix represent predicted values and column headers are used to represent
# actual values.

# Check lengths

length(status_test[, 1:12]$Status)
```

    ## [1] 153

``` r
# Check levels
status_test[, 1:12]$Status <- as.factor(status_test[, 1:12]$Status)


predictions <- predict(status_model_glm, status_test[, 1:11])
confusion_matrix <- caret::confusionMatrix(predictions, status_test[, 1:12]$Status)
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   N   Y
    ##          N  23   3
    ##          Y  25 102
    ##                                           
    ##                Accuracy : 0.817           
    ##                  95% CI : (0.7465, 0.8748)
    ##     No Information Rate : 0.6863          
    ##     P-Value [Acc > NIR] : 0.0001923       
    ##                                           
    ##                   Kappa : 0.5146          
    ##                                           
    ##  Mcnemar's Test P-Value : 7.229e-05       
    ##                                           
    ##             Sensitivity : 0.4792          
    ##             Specificity : 0.9714          
    ##          Pos Pred Value : 0.8846          
    ##          Neg Pred Value : 0.8031          
    ##              Prevalence : 0.3137          
    ##          Detection Rate : 0.1503          
    ##    Detection Prevalence : 0.1699          
    ##       Balanced Accuracy : 0.7253          
    ##                                           
    ##        'Positive' Class : N               
    ## 

``` r
### Option 3: Display a graphical confusion matrix ----

# Visualizing Confusion Matrix
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"), main = "Confusion Matrix")
```

![](Lab7a-AlgorithmSelectionForClassificationAndRegressionSubmission_files/figure-gfm/Your%20Eighth%20Code%20Chunk-1.png)<!-- -->

## \<RMSE, R Squared, and MAE\>

RMSE stands for “Root Mean Squared Error” and it is defined as the
average deviation of the predictions from the observations. R Squared
(R^2) is also known as the “coefficient of determination”. It provides a
goodness of fit measure for the predictions to the observations.

``` r
## 2.a. Load the dataset ----
data(BostonHousing)
summary(BostonHousing)
```

    ##       crim                zn             indus       chas         nox        
    ##  Min.   : 0.00632   Min.   :  0.00   Min.   : 0.46   0:471   Min.   :0.3850  
    ##  1st Qu.: 0.08205   1st Qu.:  0.00   1st Qu.: 5.19   1: 35   1st Qu.:0.4490  
    ##  Median : 0.25651   Median :  0.00   Median : 9.69           Median :0.5380  
    ##  Mean   : 3.61352   Mean   : 11.36   Mean   :11.14           Mean   :0.5547  
    ##  3rd Qu.: 3.67708   3rd Qu.: 12.50   3rd Qu.:18.10           3rd Qu.:0.6240  
    ##  Max.   :88.97620   Max.   :100.00   Max.   :27.74           Max.   :0.8710  
    ##        rm             age              dis              rad        
    ##  Min.   :3.561   Min.   :  2.90   Min.   : 1.130   Min.   : 1.000  
    ##  1st Qu.:5.886   1st Qu.: 45.02   1st Qu.: 2.100   1st Qu.: 4.000  
    ##  Median :6.208   Median : 77.50   Median : 3.207   Median : 5.000  
    ##  Mean   :6.285   Mean   : 68.57   Mean   : 3.795   Mean   : 9.549  
    ##  3rd Qu.:6.623   3rd Qu.: 94.08   3rd Qu.: 5.188   3rd Qu.:24.000  
    ##  Max.   :8.780   Max.   :100.00   Max.   :12.127   Max.   :24.000  
    ##       tax           ptratio            b              lstat      
    ##  Min.   :187.0   Min.   :12.60   Min.   :  0.32   Min.   : 1.73  
    ##  1st Qu.:279.0   1st Qu.:17.40   1st Qu.:375.38   1st Qu.: 6.95  
    ##  Median :330.0   Median :19.05   Median :391.44   Median :11.36  
    ##  Mean   :408.2   Mean   :18.46   Mean   :356.67   Mean   :12.65  
    ##  3rd Qu.:666.0   3rd Qu.:20.20   3rd Qu.:396.23   3rd Qu.:16.95  
    ##  Max.   :711.0   Max.   :22.00   Max.   :396.90   Max.   :37.97  
    ##       medv      
    ##  Min.   : 5.00  
    ##  1st Qu.:17.02  
    ##  Median :21.20  
    ##  Mean   :22.53  
    ##  3rd Qu.:25.00  
    ##  Max.   :50.00

``` r
BostonHousing_no_na <- na.omit(BostonHousing)

# For reproducibility; by ensuring that you end up with the same 'random'
# samples
set.seed(7)

# We apply simple random sampling using the base::sample function to get 10
# samples
train_index <- sample(1:dim(BostonHousing)[1], 10)
bostonhousing_train <- BostonHousing[train_index, ]
bostonhousing_test <- BostonHousing[-train_index, ]

## 2.c. Train the Model ---- We apply bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)

# We then train a linear regression model to predict the value of medv.
bostonhousing_model_lm <- train(medv ~ ., data = bostonhousing_train, na.action = na.omit,
    method = "lm", metric = "RMSE", trControl = train_control)

## 2.d. Display the Model's Performance ---- Option 1: Use the metric
## calculated by caret when training the model ---- The results show an RMSE
## value of approximately 4.3898 and an R Squared value of approximately 0.8594
## (the closer the R squared value is to 1, the better the model).
print(bostonhousing_model_lm)
```

    ## Linear Regression 
    ## 
    ## 10 samples
    ## 13 predictors
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (1000 reps) 
    ## Summary of sample sizes: 10, 10, 10, 10, 10, 10, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   68.68815  0.6348638  50.52216
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(bostonhousing_model_lm, bostonhousing_test[, 1:13])

# These are the 6 values for employment that the model has predicted:
print(predictions)
```

    ##          1          2          3          4          5          6          7 
    ## 22.8729081 16.8291226 25.1224511 18.6975233 18.2199780 12.5907463 19.9090282 
    ##          8          9         10         11         12         13         14 
    ## 15.7012764 11.3309369 16.2266554 17.3121870 16.8707315 23.9677121 20.8805233 
    ##         15         16         17         18         19         20         21 
    ## 17.9229416 21.0837618 26.5349194 17.7363032 22.1812548 18.2146976 12.0818835 
    ##         22         23         24         25         26         27         28 
    ## 16.2940816 17.0327081 13.3386104 15.0766224 14.3471833 14.9415984 16.8171751 
    ##         29         30         31         32         33         34         35 
    ## 18.8636447 21.3557593 13.6658884 15.0474244 17.4350205 13.5205999 15.8295350 
    ##         36         37         38         39         40         41         42 
    ## 13.7331012 14.3052854 17.7558649 20.5436500 30.0141052 33.9663368 29.8747933 
    ##         43         44         45         46         47         48         49 
    ## 25.1661835 25.4661631 18.6045398 17.2162112 18.0051326 10.3207484  4.2950013 
    ##         50         51         52         53         54         55         56 
    ## 11.4905228 16.5950544 14.5736841 24.6316643 21.1173094 20.2763559 30.2279795 
    ##         57         58         59         60         61         62         63 
    ## 23.7205599 26.2075130 17.7845989 13.3327257  8.6625151  5.4717255 13.2075374 
    ##         64         65         66         67         68         69         70 
    ## 19.4108589 14.9915322 26.6474768 20.9081786 17.5951236 12.9592763 15.5932553 
    ##         71         72         73         74         75         76         77 
    ## 28.3717053 23.3689265 25.7846847 27.2752001 30.9069058 24.0646537 18.9131606 
    ##         78         79         80         81         82         83         84 
    ## 23.0272667 22.0591963 22.7659772 22.9296893 15.6957327 20.2912564 16.8241944 
    ##         85         86         87         88         89         90         91 
    ## 16.6706595 16.9418585 14.7336021 13.5308846 16.7041165 21.2815715 16.3550283 
    ##         92         93         94         95         96         97         98 
    ## 14.8995087 33.2995434 36.0958349 27.8300607 15.4795844 10.2822709 22.0152711 
    ##         99        100        101        102        104        105        106 
    ## 27.2301529 19.9918397 21.7336487 23.5993552 16.4399988 16.2513518 13.0056474 
    ##        107        108        109        110        111        112        113 
    ## 13.7259869 16.8921368 17.0694734 16.4241917 22.6414719 24.0773296 16.7416006 
    ##        114        115        116        117        119        120        121 
    ## 17.4611600 20.5959872 17.6449856 22.0312882 19.9283640 20.3083214 45.6796928 
    ##        122        123        124        125        126        127        128 
    ## 44.0581630 42.2391043 40.8349363 41.1914973 43.2167242 39.4680285 36.4151777 
    ##        129        130        131        132        133        134        135 
    ## 40.8924247 36.1883707 40.9905074 40.2278817 40.5268052 37.2410805 36.2617171 
    ##        136        137        138        139        140        141        142 
    ## 40.2726303 38.4940587 41.1085437 37.1602324 39.1794577 40.1199604 31.1552526 
    ##        143        144        145        146        147        148        149 
    ## 46.8778604 47.2509472 43.9187142 51.8187353 48.4237063 44.4439410 46.5208370 
    ##        150        151        152        153        154        155        156 
    ## 49.0771872 52.2410762 46.9352766 46.4235304 49.2122113 52.5037517 54.8958316 
    ##        157        158        159        160        161        162        163 
    ## 47.0146776 39.5064485 33.1487933 54.3653693 35.6902497 44.3199803 45.0905623 
    ##        164        165        166        167        168        169        170 
    ## 49.7078819 32.9677365 34.4120992 46.2885129 35.3696140 35.4130882 36.0881094 
    ##        171        172        173        174        175        176        177 
    ## 32.6765784 32.1764913  6.2957672 12.7587448 11.7034659 22.5238752 16.3979909 
    ##        178        179        180        181        182        183        184 
    ## 13.8217816 17.4086397 19.4516342 20.3776238 13.1893592 14.6995956 10.0735100 
    ##        185        186        187        188        189        190        191 
    ##  4.5954944 11.9409081 25.9610356 23.2633222 23.6840856 26.2034379 27.2811806 
    ##        192        193        195        196        197        198        199 
    ## 24.2124358 27.9670288 28.2088986 33.1613222 30.3220960 28.6663605 29.4941141 
    ##        200        201        202        203        204        205        206 
    ## 32.5587827 33.8860708 23.9266505 37.7001286 37.3847516 38.8361946 27.2859772 
    ##        207        208        209        210        211        212        213 
    ## 24.8041260 17.5776796 21.9030646  9.8946821 15.4591705 12.3578184 21.2241028 
    ##        214        215        216        217        219        220        221 
    ## 28.7888725 26.3249246 25.7026052 28.8097035 22.6150953 25.6076808 15.2243924 
    ##        222        223        224        225        226        227        228 
    ##  9.3772609 16.5386422 14.2314683 25.6860210 27.9421910 22.6425686 17.8863149 
    ##        229        230        231        232        233        234        235 
    ## 32.4765235 24.0401535 11.8852803 19.9946697 27.0342481 26.9926595 17.4096289 
    ##        236        237        238        239        240        241        242 
    ## 13.9793665 14.8950119 20.6650904 22.8637894 19.5286928 19.3266936 12.0113764 
    ##        243        244        245        246        247        248        249 
    ## 15.7870060 23.9722366  5.5359867  6.7284444 16.3151665  9.3104558 16.0645199 
    ##        250        251        252        253        254        255        256 
    ## 23.5567692 22.8864420 23.2741773 26.8306319 35.3173355 25.0029073 25.7122365 
    ##        257        258        259        260        261        262        263 
    ## 34.2310283 40.0295955 28.4456518 25.1079260 30.7375385 31.5312758 37.0547150 
    ##        264        265        266        267        268        269        270 
    ## 29.3256203 29.0686068 23.0138649 28.9495039 35.7265052 32.5901584 18.2339920 
    ##        272        273        274        275        276        277        278 
    ## 28.2475549 22.8963976 31.7904944 29.0228042 27.8823389 29.4642709 30.2473321 
    ##        279        280        281        282        283        284        285 
    ## 27.2865685 21.6817725 22.6726669 21.6180340 23.9884008 38.3560193 35.1791988 
    ##        286        287        288        289        290        291        292 
    ## 22.2670394 23.1888774 20.6706147 18.8645722 24.5534352 31.8839994 33.8533079 
    ##        293        294        295        296        297        299        300 
    ## 31.1184466 29.7337207 24.7242558 31.1120816 26.7379938 22.7570662 29.2346659 
    ##        301        302        303        304        305        306        307 
    ## 21.4950819 21.3098735 24.5437132 27.9529249 23.6432894 16.6025187 19.6519719 
    ##        308        309        310        311        312        313        314 
    ## 16.0645806 24.7269320 21.3301505 21.4355192 26.6284563 19.3219610 22.2090871 
    ##        315        316        317        318        319        320        321 
    ## 23.3681940 19.1772159 19.6026173 20.7364322 25.6761969 25.2367879 20.9237033 
    ##        322        323        324        325        326        327        328 
    ## 20.2329009 18.6991675 12.1547118 22.9511034 27.3966460 24.0837326 19.9331287 
    ##        329        330        331        332        333        334        335 
    ## 15.2470240 19.9008172 15.8411526 23.1079080 26.2009767 21.1310690 21.0208670 
    ##        336        337        338        339        340        341        342 
    ## 19.9851344 16.9333335 14.6820833 19.8919137 17.9633419 15.5388849 24.4430513 
    ##        343        344        345        346        347        348        349 
    ## 18.6336831 24.6533959 30.6859013 12.7306615 11.2761976 29.8075960 27.3975765 
    ##        350        351        352        353        354        355        356 
    ## 24.1225859 19.3455531 20.4697800 18.8477648 24.9396894 20.0876453 22.3491384 
    ##        357        358        359        360        361        362        363 
    ## 26.2466351 28.8330209 28.2372720 28.6080072 29.3381532 27.8894176 21.0420202 
    ##        364        365        366        367        368        369        370 
    ## 25.2973985 42.9426626  6.8418737 15.7067615  0.2737006  8.2949426 20.3671800 
    ##        371        372        373        374        375        376        377 
    ## 22.4722415 16.4800602 18.6136418 10.0629275  4.4488121 26.3447625 22.9259969 
    ##        378        379        380        381        382        383        384 
    ## 23.2571736 20.0933475 18.7080008 20.9259784 21.1294881 16.5340762 16.5073201 
    ##        385        386        387        388        389        390        391 
    ##  9.5856459 14.7007543  9.6884441 13.9836520 11.7992744 15.7727156 18.3154785 
    ##        393        394        395        396        397        398        399 
    ## 13.5060655 21.7793923 19.0657515 22.5721056 22.8000735 17.7509137 13.7786515 
    ##        400        401        402        403        404        405        406 
    ## 22.0740726 18.1524014 21.2009531 21.8747465 14.5424567 16.6644763 13.5852439 
    ##        407        408        409        410        411        412        413 
    ##  3.6913295 14.1093177 10.5259911 18.0686721  8.5076239 16.7625895  2.7821408 
    ##        414        416        417        418        419        420        421 
    ##  5.7449807 20.5708507 24.9755902 14.4399474 14.0600113 30.3995888 23.4991681 
    ##        422        423        424        425        426        427        428 
    ## 21.8332663 13.3330538 17.1262874 13.8837832 17.8676496 17.4493174 21.5894371 
    ##        429        430        431        432        433        434        435 
    ## 23.4212699 21.4738210 16.4551452 18.1813193 19.0569540 25.6879450 22.4221200 
    ##        436        437        438        439        440        441        442 
    ## 27.3683451 26.2960673 23.0037866 23.7818079 20.9015337 21.6910824 25.5034361 
    ##        443        444        445        446        447        448        449 
    ## 24.0023167 25.5470925 21.7375561 26.2443395 25.4088000 24.5228416 21.8789516 
    ##        450        451        452        453        454        455        456 
    ## 23.6386872 26.9022073 25.3498705 24.0804340 29.9480321 26.2927413 26.5588857 
    ##        457        458        459        460        461        462        463 
    ## 22.5813864 23.3985414 25.2880046 23.7507058 27.0912937 25.2533650 25.5950987 
    ##        464        465        466        468        469        470        471 
    ## 25.7365145 23.8750382 24.1266656 12.7838817 15.3904308 16.6409497 15.3661002 
    ##        472        473        474        475        477        478        479 
    ## 11.3185659 18.8503896 26.1626532  8.5459278 18.2676697  9.0927181 15.4134643 
    ##        480        481        482        483        484        485        486 
    ## 17.0471185 15.8138386 17.4703964 19.1805198 16.9104946 20.9650791 22.0436048 
    ##        487        488        489        490        491        492        493 
    ## 15.7646429 19.1788717 41.4252712 40.1802596 38.0512138 43.9121489 46.5571532 
    ##        494        495        496        497        498        499        500 
    ## 24.4239751 27.9051185 28.5269907 18.8567522 21.9694653 24.5275054 20.0510013 
    ##        501        502        503        504        505        506 
    ## 22.0247357 34.2071857 29.7176403 32.9977912 32.0189749 28.3407871 
    ## attr(,"non-estim")
    ##   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
    ##   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
    ##  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
    ##  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
    ##  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
    ##  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
    ##  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
    ##  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
    ##  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
    ##  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
    ## 101 102 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119 120 121 122 
    ## 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
    ## 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 
    ## 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
    ## 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 
    ## 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 
    ## 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 
    ## 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 
    ## 183 184 185 186 187 188 189 190 191 192 193 195 196 197 198 199 200 201 202 203 
    ## 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 
    ## 204 205 206 207 208 209 210 211 212 213 214 215 216 217 219 220 221 222 223 224 
    ## 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 
    ## 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 
    ## 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 
    ## 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 
    ## 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 
    ## 265 266 267 268 269 270 272 273 274 275 276 277 278 279 280 281 282 283 284 285 
    ## 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 
    ## 286 287 288 289 290 291 292 293 294 295 296 297 299 300 301 302 303 304 305 306 
    ## 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 
    ## 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 
    ## 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 
    ## 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 
    ## 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 
    ## 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 
    ## 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 
    ## 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 
    ## 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 
    ## 387 388 389 390 391 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 
    ## 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 
    ## 408 409 410 411 412 413 414 416 417 418 419 420 421 422 423 424 425 426 427 428 
    ## 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 
    ## 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 
    ## 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 
    ## 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 468 469 
    ## 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 
    ## 470 471 472 473 474 475 477 478 479 480 481 482 483 484 485 486 487 488 489 490 
    ## 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 
    ## 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 
    ## 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496

``` r
#### RMSE ----
rmse <- sqrt(mean((bostonhousing_test$medv - predictions)^2))
print(paste("RMSE =", rmse))
```

    ## [1] "RMSE = 11.7787223672508"

``` r
#### SSR ---- SSR is the sum of squared residuals (the sum of squared
#### differences between observed and predicted values)
ssr <- sum((bostonhousing_test$medv - predictions)^2)
print(paste("SSR =", ssr))
```

    ## [1] "SSR = 68814.197099968"

``` r
#### SST ---- SST is the total sum of squares (the sum of squared differences
#### between observed values and their mean)
sst <- sum((bostonhousing_test$medv - mean(bostonhousing_test$medv))^2)
print(paste("SST =", sst))
```

    ## [1] "SST = 42230.6680443548"

``` r
#### R Squared ---- We then use SSR and SST to compute the value of R squared
r_squared <- 1 - (ssr/sst)
print(paste("R Squared =", r_squared))
```

    ## [1] "R Squared = -0.629483981349584"

``` r
#### MAE ---- MAE measures the average absolute differences between the
#### predicted and actual values in a dataset. MAE is useful for assessing how
#### close the model's predictions are to the actual values.

absolute_errors <- abs(predictions - bostonhousing_test$medv)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))
```

    ## [1] "MAE = 8.53538821380395"

## \<Area Under ROC Curve\>

Area Under Receiver Operating Characteristic Curve (AUROC) or simply
“Area Under Curve (AUC)” or “ROC” represents a model’s ability to
discriminate between two classes.

``` r
status_freq <- train_imputed$Status
cbind(frequency = table(status_freq), percentage = prop.table(table(status_freq)) *
    100)
```

    ##   frequency percentage
    ## N       192   31.27036
    ## Y       422   68.72964

``` r
## 3.c. Split the dataset ---- Define an 80:20 train:test data split of the
## dataset.
train_index <- createDataPartition(train_imputed$Status, p = 0.8, list = FALSE)
status_train <- train_imputed[train_index, ]
status_test <- train_imputed[-train_index, ]

## 3.d. Train the Model ---- We apply the 10-fold cross validation resampling
## method
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# We then train a k Nearest Neighbours Model to predict the value of Status

set.seed(7)
status_model_knn <- train(Status ~ ., data = status_train, method = "knn", metric = "ROC",
    trControl = train_control)
## 3.e. Display the Model's Performance ---- Option 1: Use the metric
## calculated by caret when training the model ---- The results show a ROC
## value of approximately 0.76 (the closer to 1, the higher the prediction
## accuracy) when the parameter k = 9 (9 nearest neighbours).

print(status_model_knn)
```

    ## k-Nearest Neighbors 
    ## 
    ## 492 samples
    ##  11 predictor
    ##   2 classes: 'N', 'Y' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 442, 442, 442, 444, 443, 443, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  ROC        Sens       Spec     
    ##   5  0.5003082  0.2020833  0.8524064
    ##   7  0.5208805  0.1816667  0.8818182
    ##   9  0.5393546  0.1691667  0.8966132
    ## 
    ## ROC was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

``` r
#### AUC ---- The type = 'prob' argument specifies that you want to obtain
#### class probabilities as the output of the prediction instead of class
#### labels.
predictions <- predict(status_model_knn, status_test[, 1:12], type = "prob")

# These are the class probability values for Status that the model has
# predicted:
print(predictions)
```

    ##             N         Y
    ## 1   0.4444444 0.5555556
    ## 2   0.5555556 0.4444444
    ## 3   0.4444444 0.5555556
    ## 4   0.4444444 0.5555556
    ## 5   0.1111111 0.8888889
    ## 6   0.3333333 0.6666667
    ## 7   0.1111111 0.8888889
    ## 8   0.2222222 0.7777778
    ## 9   0.3333333 0.6666667
    ## 10  0.3000000 0.7000000
    ## 11  0.3333333 0.6666667
    ## 12  0.3333333 0.6666667
    ## 13  0.1111111 0.8888889
    ## 14  0.3333333 0.6666667
    ## 15  0.1111111 0.8888889
    ## 16  0.2222222 0.7777778
    ## 17  0.1111111 0.8888889
    ## 18  0.2222222 0.7777778
    ## 19  0.2222222 0.7777778
    ## 20  0.2222222 0.7777778
    ## 21  0.3333333 0.6666667
    ## 22  0.1111111 0.8888889
    ## 23  0.4444444 0.5555556
    ## 24  0.3333333 0.6666667
    ## 25  0.1111111 0.8888889
    ## 26  0.1111111 0.8888889
    ## 27  0.3333333 0.6666667
    ## 28  0.4444444 0.5555556
    ## 29  0.1111111 0.8888889
    ## 30  0.2222222 0.7777778
    ## 31  0.1111111 0.8888889
    ## 32  0.4444444 0.5555556
    ## 33  0.2222222 0.7777778
    ## 34  0.3000000 0.7000000
    ## 35  0.0000000 1.0000000
    ## 36  0.4444444 0.5555556
    ## 37  0.2222222 0.7777778
    ## 38  0.3000000 0.7000000
    ## 39  0.1111111 0.8888889
    ## 40  0.4444444 0.5555556
    ## 41  0.2222222 0.7777778
    ## 42  0.1111111 0.8888889
    ## 43  0.3333333 0.6666667
    ## 44  0.1111111 0.8888889
    ## 45  0.2222222 0.7777778
    ## 46  0.2222222 0.7777778
    ## 47  0.2222222 0.7777778
    ## 48  0.0000000 1.0000000
    ## 49  0.4444444 0.5555556
    ## 50  0.3333333 0.6666667
    ## 51  0.0000000 1.0000000
    ## 52  0.4444444 0.5555556
    ## 53  0.3333333 0.6666667
    ## 54  0.2222222 0.7777778
    ## 55  0.1111111 0.8888889
    ## 56  0.1111111 0.8888889
    ## 57  0.2222222 0.7777778
    ## 58  0.3333333 0.6666667
    ## 59  0.4444444 0.5555556
    ## 60  0.2222222 0.7777778
    ## 61  0.0000000 1.0000000
    ## 62  0.1111111 0.8888889
    ## 63  0.2222222 0.7777778
    ## 64  0.2222222 0.7777778
    ## 65  0.3333333 0.6666667
    ## 66  0.2222222 0.7777778
    ## 67  0.1111111 0.8888889
    ## 68  0.1111111 0.8888889
    ## 69  0.3333333 0.6666667
    ## 70  0.4444444 0.5555556
    ## 71  0.0000000 1.0000000
    ## 72  0.2222222 0.7777778
    ## 73  0.2222222 0.7777778
    ## 74  0.3333333 0.6666667
    ## 75  0.2222222 0.7777778
    ## 76  0.2222222 0.7777778
    ## 77  0.3333333 0.6666667
    ## 78  0.3333333 0.6666667
    ## 79  0.1111111 0.8888889
    ## 80  0.2222222 0.7777778
    ## 81  0.4444444 0.5555556
    ## 82  0.1111111 0.8888889
    ## 83  0.3333333 0.6666667
    ## 84  0.7777778 0.2222222
    ## 85  0.4444444 0.5555556
    ## 86  0.5555556 0.4444444
    ## 87  0.3000000 0.7000000
    ## 88  0.3333333 0.6666667
    ## 89  0.2222222 0.7777778
    ## 90  0.0000000 1.0000000
    ## 91  0.0000000 1.0000000
    ## 92  0.4444444 0.5555556
    ## 93  0.1111111 0.8888889
    ## 94  0.2222222 0.7777778
    ## 95  0.4444444 0.5555556
    ## 96  0.2222222 0.7777778
    ## 97  0.4444444 0.5555556
    ## 98  0.3333333 0.6666667
    ## 99  0.3333333 0.6666667
    ## 100 0.3333333 0.6666667
    ## 101 0.2222222 0.7777778
    ## 102 0.2222222 0.7777778
    ## 103 0.0000000 1.0000000
    ## 104 0.4444444 0.5555556
    ## 105 0.4444444 0.5555556
    ## 106 0.3333333 0.6666667
    ## 107 0.2222222 0.7777778
    ## 108 0.3333333 0.6666667
    ## 109 0.6666667 0.3333333
    ## 110 0.0000000 1.0000000
    ## 111 0.0000000 1.0000000
    ## 112 0.1111111 0.8888889
    ## 113 0.5555556 0.4444444
    ## 114 0.2222222 0.7777778
    ## 115 0.4444444 0.5555556
    ## 116 0.2000000 0.8000000
    ## 117 0.4444444 0.5555556
    ## 118 0.2222222 0.7777778
    ## 119 0.3000000 0.7000000
    ## 120 0.2222222 0.7777778
    ## 121 0.1111111 0.8888889
    ## 122 0.3333333 0.6666667

``` r
roc_curve <- roc(status_test$Status, predictions$N)
```

    ## Setting levels: control = N, case = Y

    ## Setting direction: controls < cases

``` r
# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE, print.auc.x = 0.6,
    print.auc.y = 0.6, col = "blue", lwd = 2.5)
```

![](Lab7a-AlgorithmSelectionForClassificationAndRegressionSubmission_files/figure-gfm/Your%20Tenth%20Code%20Chunk-1.png)<!-- -->

## \<Logarithmic Loss \>

Logarithmic Loss (LogLoss) is an evaluation metric commonly used for
assessing the performance of classification models, especially when the
model provides probability estimates for each class. LogLoss measures
how well the predicted probabilities align with the true binary
outcomes.

``` r
# We apply the 5-fold repeated cross validation resampling method with 3
# repeats
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, classProbs = TRUE,
    summaryFunction = mnLogLoss)
set.seed(7)
# This creates a CART model. One of the parameters used by a CART model is
# 'cp'.  'cp' refers to the 'complexity parameter'. It is used to impose a
# penalty to the tree for having too many splits. The default value is 0.01.
status_model_cart <- train(Status ~ ., data = train_imputed, method = "rpart", metric = "logLoss",
    trControl = train_control)

## 4.c. Display the Model's Performance ---- Option 1: Use the metric
## calculated by caret when training the model ---- The results show that a cp
## value of ≈ 0 resulted in the lowest LogLoss value. The lowest logLoss value
## is ≈ 0.46.
print(status_model_cart)
```

    ## CART 
    ## 
    ## 614 samples
    ##  11 predictor
    ##   2 classes: 'N', 'Y' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 491, 491, 492, 492, 490, 491, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp           logLoss  
    ##   0.002604167  0.6476700
    ##   0.006944444  0.5031450
    ##   0.390625000  0.5692996
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.006944444.

**etc.** as per the lab submission requirements. Be neat and communicate
in a clear and logical manner.
