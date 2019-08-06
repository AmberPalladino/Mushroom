# Amber Palladino
# R script for HarvardX Data Science Capstone - Choose Your Own Project

# Data used is from The Audobon Society Field Guide; 
# mushrooms described in terms of physical characteristics; 
# classification: poisonous or edible

# Mushroom dataset: Description
# https://archive.ics.uci.edu/ml/datasets/Mushroom
# Mushroom dataset: Data Download
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data

# Project goal is to create a machine learning algorithm to successfully classify
# mushrooms as poisonous or edible based on their physical characteristics

# Install required packages, if needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

# Download mushroom data from the University of California - Irvine database
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", dl)

# Read downloaded .data file into an R data frame and add column names
mushrooms <- read.csv(dl, header=FALSE, sep=",", col.names = c("class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color", "ring_number", "ring_type", "spore_print_color", "population", "habitat"))

# Data exploration and visualization
head(mushrooms)
class(mushrooms)
dim(mushrooms)
summary(mushrooms)

# The attribute being predicted by this algorithm is labeled "class", with values of "e" ("edible") or "p" ("poisonous")
plot(mushrooms$class, main = "Distribution by Class \n (Edible versus Poisonous)")
sum(mushrooms$class == "e") / length(mushrooms$class)
# Approximately 52% of the mushrooms in the dataset are edible, and roughly 48% are poisonous

# Set seed to ensure that partition results will be the same every time the code is executed
# If using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1, sample.kind="Rounding")

# Partition data into training and test sets
# Test set will be 10% of mushroom data
test_index <- createDataPartition(y = mushrooms$class, times = 1, p = 0.1, list = FALSE)
mtrain <- mushrooms[-test_index,]
mtest <- mushrooms[test_index,]

# Create train and test subsets within the training data for algorithm evaluation and tuning
# Evaluation test subset will be 20% of training data
test_index_subset <- createDataPartition(y = mtrain$class, times = 1, p = 0.2, list = F)
mtrain_subset <- mtrain[-test_index,]
mtest_subset <- mtrain[test_index,]

# Remove temporary objects
rm(dl, test_index, test_index_subset)

# Use rpart to generate a classification tree
rp_fit <- rpart(class ~ .,
                data = mtrain_subset)

# Visualize classification tree structure and summary of nodes generated
printcp(rp_fit)
summary(rp_fit)
plot(rp_fit, main = "Mushroom Classification Tree")
text(rp_fit, cex = 0.8)

# Determine lowest CP to create pruned tree
min_cp <- rp_fit$cptable[which.min(rp_fit$cptable[,"xerror"]), "CP"]
pruned <- prune(rp_fit, cp = min_cp)

# Generate classification predictions on training data and view as confusion matrix table
table(mtrain_subset$class, predict(pruned, type = "class"), dnn = c("Actual", "Predicted"))

# Can results be improved using a random forest of trees?
# Use RandomForest to generate a series of classification trees
rf_fit <- randomForest(class ~ .,
                       data = mtrain_subset)

# View RandomForest results
print(rf_fit)

# View importance of each predictor
importance(rf_fit)

# Create predictions for the training subset and determine accuracy of model
pred <- predict(rf_fit, newdata = mtrain_subset)
confusionMatrix(pred, mtrain_subset$class)$overall["Accuracy"]

# Create predictions for the test subset and determine accuracy of model to ensure the model has not been overfit
confusionMatrix(predict(rf_fit, mtest_subset), mtest_subset$class)$overall["Accuracy"]

# Execute final prediction, using full training dataset to train model
final_rf_fit <- randomForest(class ~ .,
                             data = mtrain)

# Generate predictions for test data using final model and view results and accuracy
confusionMatrix(predict(final_rf_fit, mtest), mtest$class)

