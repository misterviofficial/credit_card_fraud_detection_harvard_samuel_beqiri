# Credit Card Fraud project for HarvardX chose my own project.
#Samuel_Beqiri

# Initial set up.
# The following packages will be installed.
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gbm)) install.packages("gbm")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(xgboost)) install.packages("xgboost")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(corrplot)) install.packages("corrplot")

# Load the following packages using the library() function.
library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(gbm)
library(caret)
library(xgboost)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(corrplot)

# The data available in kaggle as the most rated datasete.
# The dataset (as a .csv files) can be downloaded at the following link:
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# download card's dataset to your pc.

# Loading the dataset as a .csv file on local system.
# I will save it as mydataset for the credit card data set.
mydataset <- read.csv("creditcard.csv")

#we will view the dimension of the dataset
dim(mydataset)

# We will view the first information for the dataset.
head(mydataset)

# We will make a table of the dimensions of the data set
# and display the table.
data.frame("Length" = nrow(mydataset), "Columns" = ncol(mydataset)) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 12,
                full_width = FALSE)

# We want to see how many transactions are fraud compared to 
# Normal, 0 is defined as a Normal transaction, and not 0 is defined as a fraudulent transaction.

Fraudornot <- data.frame(mydataset)
Fraudornot$Class = ifelse(mydataset$Class == 0, 'Normal', 'Fraud') %>% as.factor()

# To see the data, we plot a bar graph of the frequency of fraud 
# verses Normal credit card transactions.
# We see that the vast majority of transactions are Normal
Fraudornot %>%
  ggplot(aes(Class)) +
  theme_minimal()  +
  geom_bar(fill = "black") +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Transaction tyeps",
       x = "type",
       y = "Frequency")

#there is no missing values, and that why next function will return false.
anyNA(mydataset)

#summary of each variable in the dataset.
summary(mydataset)

# how many dollar amounts of fraud.
# Here we plot all the fraudulent transaction by amount.
# This plot shows a massive skew toward transactions under $100
mydataset[mydataset$Class == 1,] %>%
  ggplot(aes(Amount)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 50, fill = "black") +
  labs(title = "Distribution of fraud amounts",
       x = "Amount in Dollars",
       y = "Frequency")

# To further investigate this, we make a table of the 10 most common
# fraudulent transactions. 
# By far, $1 is the most fraudulent transaction. It is also interesting
# to note that a $0 transaction and a $99.99 transaction are tied for second
# in most common fraudulent transactions. 
mydataset[mydataset$Class == 1,] %>%
  group_by(Amount) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)

# We can also investigate what are the most common valid
# transactions in the dataset.
# An interesting observation is that $1 is the most common
# fraudulent and valid transaction.
# In fact ~0.83% of $1 transactions are fraud, compared to
# ~0.17% - almost five times higher than other transactions
# in the data set.
# $99.99 is number 98 on the list of valid transactions with 303
# transactions, but tied for second of fraudulent transactions with
# 27. This means that 27% of $99.99 transactions are fraudulent.
mydataset[mydataset$Class == 0,] %>%
  group_by(Amount) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)

# Here we plot a summary of the mean and median transaction for 
# valid and fraudulent transactions.
mydataset %>% 
  group_by(Class) %>% 
  summarize(mean(Amount), median(Amount))

# We can plot a distribution of valid transactions over time.
# This plot has a clear episodic distribution. This makes sense since 
# a day has 86,400 seconds, which is the approximate period of this
# distribution. The punchline is that most transactions occur during
# the day, while fewer transactions occur at night. 
# There is a clear spike of outlier transactions near the trough of
# the graph. We surmise that these spikes correlate to automated
# transactions that are processed a little before the close of midnight
# or shortly after midnight. An example of automated transactions 
# would be monthly recurring bills set to autopay.
mydataset[mydataset$Class == 0,] %>%
  ggplot(aes(Time)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 100, fill = "black") +
  labs(title = "Valid Transacations Distribution",
       x = "Time [seconds]",
       y = "Frequency")

# Similarly, to the distribution of valid transactions, we can plot
# the distribution of fraudulent transactions over time. 
# The lack of any clear episodic distribution indicates that
# fraud can occur at any time.
mydataset[mydataset$Class == 1,] %>%
  ggplot(aes(Time)) + 
  theme_minimal()  +
  geom_histogram(binwidth = 25, fill = "black") +
  labs(title = "Fraudulent Transactions Distribution",
       x = "Time [seconds]",
       y = "Frequency")

# To note: Without performing Fourier analysis (such as a Fast
# Fourier Transform) on this data, we do not know with certainty 
# that fraudulent transactions are non-episodic. This analysis is
# beyond the scope of this project, and the frequency distribution
# plotted above will suffice to show that fraudulent transactions
# are not episodic and can occur at any point in time.

# We want to calculate the correlation between the variables
# and graph them. We first design a correlation matrix.

# We obtain the lower triangle of the correlation matrix.
get_lower_triangle<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# We obtain the upper triangle of the correlation matrix.
get_upper_triangle <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

# We then triangulate the upper and lower portions
# to create a correlation graph.
reorder_cormat <- function(cormat){
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

corr_matrix <- round(cor(mydataset),2)
corr_matrix <- reorder_cormat(corr_matrix)

# Here is a matrix of the correlation between the
# 31 distinct variables.
corr_matrix %>%
  head(n=31) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)

upper_triangle <- get_upper_triangle(corr_matrix)
melted_corr_matrix <- melt(upper_triangle, na.rm = TRUE)

# Further, we can plot the correlation.
# Notice how all the variables V1-V28 have very low correlation 
# coefficients among each other, and especially low correlation 
# with the 'Class' feature. This was already expected since the 
# data was processed using PCA.
ggplot(melted_corr_matrix, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "yellow4") +
  scale_fill_gradient2(low = "yellowgreen", high = "yellow", mid = "yellow3", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Variable Correlation") +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 5, hjust = 1), axis.text.y = element_text(size = 5),                    axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.ticks = element_blank()) +
  coord_fixed()

# We established that fraud does not appear to coincide with
# a specific time of day, so the "Time" variable will be 
# removed from the dataset.
mydataset$Class <- as.factor(mydataset$Class)
mydataset <- mydataset %>% select(-Time)

# To verify the variable Time has been removed,
# we can view the first 10 entries in full table format.
mydataset %>%
  head(n=10) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)

#### NAIVE MODEL ####

# Set seed for reproducibility.
set.seed(13)

# We need to create a training data set, a test dataset,
# and a cross validation data set.
# Here we partition the data.
train_index <- createDataPartition(
  y = mydataset$Class, 
  p = .6, 
  list = F)

# Our training set.
train <- mydataset[train_index,]

# Temporary test dataset.
test_cv <- mydataset[-train_index,]

# We partition the test dataset.
test_index <- createDataPartition(
  y = test_cv$Class, 
  p = .5, 
  list = F)

# The partitioned test dataset is split between
# a test set and a cross validation set.
test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

# We remove the temporary files to create our datasets.
rm(train_index, test_index, test_cv)

# Here we will create a Navie Model that will serve as a 
# baseline. Here we will make the simple prediction that
# every transaction is a valid transaction and that there
# are not fraudulent transaction.

# Copy the mydataset dataframe to make the necessary
# changes for the baseline model. 
naive_model <- data.frame(mydataset)

# We now define all transactions as valid by defining
# all entries in the class set as valid.
naive_model$Class = factor(0, c(0,1))

# We then make the prediction of that all entries 
# are valid transactions.
pred <- prediction(
  as.numeric(as.character(naive_model$Class)),
  as.numeric(as.character(mydataset$Class)))

# We need to compute the Area Under the Curve (AUC)
# and the Area Under the Precision-Recall Curve (AUPRC).
auc_val_naive <- performance(pred, "auc")
auc_plot_naive <- performance(pred, 'sens', 'spec')
auprc_plot_naive <- performance(pred, "prec", "rec")

# We plot the AUC for the Naive Model.
# As expected, we obtain an area under the curve of 0.5.
plot(auc_plot_naive, 
     main=paste("AUC:", 
                auc_val_naive@y.values[[1]]))

# We plot the AUPRC for the Naive Model.
# Since the recall and precision are both zero, there
# is no value for the AUPRC.
plot(auprc_plot_naive, main="AUPRC: 0")


# We will create a dataframe to contain our results for AUC and AUPRC
# for the different models tested.
# Here we add our results for the Naive Model.
results <- data.frame(
  Model = "Naive", 
  AUC = auc_val_naive@y.values[[1]],
  AUPRC = 0)

# Our results are displayed in a table format.
results %>% 
  kable() %>%
  kable_styling(
    bootstrap_options = 
      c("striped", "hover", "condensed", "responsive"),
    position = "center",
    font_size = 14,
    full_width = FALSE) 

#### Naive Bayes Model ####

# Set seed for reproducibility.
set.seed(13)

# For the Naive Bayes Model, we build the model with the class
# as the target and with the remaining variables are predictors.

# We start with our naive model and define the target and
# the predictors.
naive_model <- naiveBayes(Class ~ ., data = train, laplace=1)

# We then make the prediction based on our modified dataset.
predictions <- predict(naive_model, newdata=test)

# We need to compute the Area Under the Curve (AUC)
# and the Area Under the Precision-Recall Curve (AUPRC).
pred <- prediction(as.numeric(predictions), test$Class)
auc_val_naive <- performance(pred, "auc")
auc_plot_naive <- performance(pred, 'sens', 'spec')
auprc_plot_naive <- performance(pred, "prec", "rec")

# We apply the model to our test set.
auprc_val_naive <- pr.curve(
  scores.class0 = predictions[test$Class == 1], 
  scores.class1 = predictions[test$Class == 0],
  curve = T,  
  dg.compute = T)

# We plot our curves for the Naive Bayes Model.
plot(auc_plot_naive, main=paste("AUC:", auc_val_naive@y.values[[1]]))
plot(auprc_plot_naive, main=paste("AUPRC:", auprc_val_naive$auc.integral))
plot(auprc_val_naive)

# Here we add our results for the Naive Bayes Model.
results <- results %>% 
  add_row(
    Model = "Naive Bayes", 
    AUC = auc_val_naive@y.values[[1]],
    AUPRC = auprc_val_naive$auc.integral)

# Our results are displayed in a table format with previous results.
results %>%
  kable() %>%
  kable_styling(bootstrap_options = 
                  c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)

#### K-Nearest Neighbors (KNN) Model ####

# Set seed for reproducibility.
set.seed(13)

# Our next approach will be the K-Nearest Neighbor Model.
# This is building off of our previous model, the Naive Bayes
# Model, where we specify that Class is the target and all
# other variables are predictors. For this model we set k=5.
# Training this model takes a little bit of time.
knn_model <- knn(train[,-30], 
                 test[,-30], 
                 train$Class, 
                 k=5, 
                 prob = TRUE)

# We then make the prediction based on our modified dataset.
pred <- prediction(
  as.numeric(as.character(knn_model)),
  as.numeric(as.character(test$Class)))

# We need to compute the Area Under the Curve (AUC)
# and the Area Under the Precision-Recall Curve (AUPRC).
auc_val_knn <- performance(pred, "auc")
auc_plot_knn <- performance(pred, 'sens', 'spec')
auprc_plot_knn <- performance(pred, "prec", "rec")

# We apply the model to our test set.
auprc_val_knn <- pr.curve(
  scores.class0 = knn_model[test$Class == 1], 
  scores.class1 = knn_model[test$Class == 0],
  curve = T,  
  dg.compute = T)

# We plot our curves for the KNN Model.
plot(auc_plot_knn, main=paste("AUC:", auc_val_knn@y.values[[1]]))
plot(auprc_plot_knn, main=paste("AUPRC:", auprc_val_knn$auc.integral))
plot(auprc_val_knn)

# Here we add our results for the KNN Model.
results <- results %>% 
  add_row(
    Model = "K-Nearest Neighbors", 
    AUC = auc_val_knn@y.values[[1]],
    AUPRC = auprc_val_knn$auc.integral)

# Our results are displayed in a table format with previous results.
results %>%
  kable() %>%
  kable_styling(bootstrap_options = 
                  c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE) 

#### Random Forest Model ####

# Set seed for reproducibility.
set.seed(13)

# Our next approach will be the Random Forest Model.
# As with the two previous models, we specify that Class 
# is the target and all other variables being predictors.
# For the Random Forest Model, we will define the number
# of trees to be 500.
# This takes a while to train the model.
rf_model <- randomForest(Class ~ ., data = train, ntree = 500)

# We then make the prediction based on our modified dataset.
predictions <- predict(rf_model, newdata=test)

pred <- prediction(
  as.numeric(as.character(predictions)),
  as.numeric(as.character(test$Class)))

# We need to compute the Area Under the Curve (AUC)
# and the Area Under the Precision-Recall Curve (AUPRC).
auc_val_rf <- performance(pred, "auc")
auc_plot_rf <- performance(pred, 'sens', 'spec')
auprc_plot_rf <- performance(pred, "prec", "rec", 
                             curve = T,  
                             dg.compute = T)
auprc_val_rf <- pr.curve(scores.class0 = predictions[test$Class == 1], 
                         scores.class1 = predictions[test$Class == 0],
                         curve = T,  
                         dg.compute = T)

# We plot our curves for the Random Forest Model.
plot(auc_plot_rf, main=paste("AUC:", auc_val_rf@y.values[[1]]))
plot(auprc_plot_rf, main=paste("AUPRC:", auprc_val_rf$auc.integral))
plot(auprc_val_rf)

# Here we add our results for the Random Forest Model.
results <- results %>% add_row(
  Model = "Random Forest",
  AUC = auc_val_rf@y.values[[1]],
  AUPRC = auprc_val_rf$auc.integral)

# Our results are displayed in a table format with previous results.
results %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 14,
                full_width = FALSE)
