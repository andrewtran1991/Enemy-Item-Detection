# LDA
set.seed(33)

# Load in the data
df <- bank_clean2 %>%
  select(Surpass_Id, Item, text, Whole_item,new_keywords_tfidf) %>%
  rename(Keywords = new_keywords_tfidf)%>%
  mutate(Surpass_Id=as.character(Surpass_Id))

  # stop_words object combines the SMART, Snowball, and Onix stop word sets
stoplist <- stop_words
enemy_list <- Surpass_bank %>% select(Surpass_Id, Enemies) %>% 
  mutate(Surpass_Id = as.character(Surpass_Id)) %>%
  filter(!is.na(Enemies)) %>%
  filter(Surpass_Id %in% df$Surpass_Id)

# Drop stop words (a, the, because, etc.)
# stop_words object combines the SMART, Snowball, and Onix stop word sets
item_text <- df %>%
  mutate(Surpass_Id = as.factor(Surpass_Id)) %>%
  mutate(linenumber = row_number()) %>%
  unnest_tokens(word, text) %>%
  count(Surpass_Id, word) %>%
  anti_join(stop_words) # Drop the stop words (a, the, because, etc.); The stop_words object combines the SMART, Snowball, and Onix stop word sets.

# Clean remaining words: stemming, punctuation, whitespace removal
item_text <- item_text %>%
  filter(word != "") %>%
  mutate(word = stripWhitespace(word)) %>% # Whitespace
  mutate(word = removePunctuation(word,
                                  preserve_intra_word_contractions = TRUE,
                                  preserve_intra_word_dases = TRUE)) %>% # Punctuation
  mutate(word = wordStem(word)) # Porter stemming


# Create Document-Term Matrix for passing to the model in the next steps
item_DTM <- item_text %>%
  cast_dtm(Surpass_Id, word, n)

# In the following two lines we remove any documents with no words
rowTotals <- apply(item_DTM, 1, sum) # Find the # of words in each Document
item_DTM <- item_DTM[rowTotals > 0, ]  # Remove all documents without words

cat("itemText_DTM has", table(rowTotals < 1)[2], "empty documents as a result of stripping stop words.")

item_DTM$dimnames

# Write this to the LDA folder, and we'll pick up there in the next script
# write_rds(item_DTM, paste0(path_output, "item_DTM.rds"))

full_data  <- item_DTM # Make a copy of the DTM to pass to the model
n <- nrow(full_data) # How many documents in the set?

# 5-fold cross-validation, different numbers of topics 
folds <- 5 # Specify number of folds
splitfolds <- sample(1:folds, n, replace = TRUE) # Split the data into sets
candidate_k <- seq(from = 10, to = 150, by = 10) # What topic #s do we want to test?
alphaPrior <- 1/candidate_k # Rule of thumb: alpha starting value of 1/# of topics
estimation_method <- "Gibbs" # "Gibbs" or "VEM"
burnin = 1000
iter = 1000
keep = 50

# Set up a cluster for parallel processing
cluster <- makeCluster(detectCores(logical = TRUE) - 1) # Leave one core free
registerDoParallel(cluster)

# Load the needed R package on all the parallel sessions
clusterEvalQ(cluster, {
  library(topicmodels)
  library(ldatuning)
})

# export all the needed R objects to the parallel sessions
clusterExport(cluster, c("full_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k", "alphaPrior", "estimation_method"))

# we parallelize by the different number of topics.  A processor is allocated a value
# of k, and does the cross-validation serially. This is because it is assumed there
# are more candidate values of k than there are cross-validation folds, hence it
# will be more efficient to parallelise

system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    alphaPrior <- 1/k
    
    # results_1k <- matrix(0, nrow = folds, ncol = 2)
    results_1k <- matrix(0, nrow = folds, ncol = 6)
    # colnames(results_1k) <- c("Topics", "Perplexity")
    colnames(results_1k) <- c("Topics", "Split", "Perplexity", "Griffiths2004", "CaoJuan2009", "Arun2010")
    for(i in 1:folds){
      train_set <- full_data[splitfolds != i , ]
      valid_set <- full_data[splitfolds == i, ]
      
      
      # Here we check to see which estimation method we're trying
      if(estimation_method == "VEM"){
        
        # VEM approach
        fitted <- LDA(train_set,
                      k = candidate_k[j],
                      method = "VEM",
                      control = list(seed = 17,
                                     estimate.alpha = TRUE,
                                     alpha = alphaPrior,
                                     estimate.beta = TRUE),
                      mc.cores = 2L)
        results_1k[i,1:3] <- c(k, i, perplexity(fitted, newdata = valid_set))
        
        
        # VEM approach
        metrics <- FindTopicsNumber(train_set,
                                    topics = candidate_k[j],
                                    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
                                    method = "VEM",
                                    control = list(seed = 17,
                                                   estimate.alpha = TRUE,
                                                   alpha = alphaPrior,
                                                   estimate.beta = TRUE),
                                    mc.cores = 2L
        )
        results_1k[i,5:6] <- c(as.matrix(metrics[1,2:3])) 
        
      } else {
        
        # Gibbs
        fitted <- LDA(train_set,
                      k = candidate_k[j],
                      method = "Gibbs",
                      control = list(seed = 17,
                                     burnin = burnin,
                                     iter = iter,
                                     keep = keep,
                                     alpha = alphaPrior),
                      mc.cores = 2L,
                      verbose = FALSE)
        
        results_1k[i,1:3] <- c(k, i, perplexity(fitted, newdata = valid_set))
        
        # Gibbs approach
        metrics <- FindTopicsNumber(train_set,
                                    topics = candidate_k[j],
                                    # metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
                                    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
                                    method = "Gibbs",
                                    control = list(seed = 17,
                                                   burnin = burnin,
                                                   iter = iter,
                                                   keep = keep,
                                                   alpha = alphaPrior),
                                    mc.cores = 2L,
                                    verbose = FALSE)
        results_1k[i,4:6] <- c(as.matrix(metrics[1,2:4]))
        
      }
    }
    return(results_1k)
  }
})

stopCluster(cluster)

results_df <- as.data.frame(results)

if(estimation_method == "VEM"){
  saveRDS(results_df, file = "1_Fit_Statistics_VEM.RData") # Save Fist Statistics
  write.xlsx(results_df, file = "1_Fit_Statistics_VEM.xlsx")
} else {
  saveRDS(results_df, file = "1_Fit_Statistics_Gibbs.RData") # Save Fist Statistics
  write.xlsx(results_df, file = "1_Fit_Statistics_Gibbs.xlsx")
}

# results_df <- read_rds(paste0(path_output, "1_Fit_Statistics_Gibbs.RData")) 

#  Plot the fit measures  ----
#  To make a determination about number of topics, plot the fit measure

# Griffiths is a metric we seek to maximize, but the rest we want to minimize
# Flip Griffiths so we can more clearly see a common low point @ ideal topic #
values <- results_df %>%
  mutate(Split = as.factor(Split),
         Griffiths2004 = Griffiths2004 * -1) %>% 
  print()

# Function for standardizing results
scaleThis <- function(x){
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

# Standardize fit indices
values_standardized <- values %>%
  mutate(Griffiths2004 = scaleThis(Griffiths2004),
         Perplexity = scaleThis(Perplexity),
         CaoJuan2009 = scaleThis(CaoJuan2009),
         Arun2010 = scaleThis(Arun2010))

values_standardized_Optimum <- values_standardized %>%
  group_by(Topics) %>%
  summarise(Perplexity = mean(Perplexity),
            Griffiths2004 = mean(Griffiths2004),
            CaoJuan2009 = mean(CaoJuan2009),
            Arun2010 = mean(Arun2010)) %>%
  mutate(Optimum = ((Perplexity) + (CaoJuan2009) + (Arun2010) + Griffiths2004)/4,
         Split = as.factor("Mean")) %>%
  select(Topics, Split, Perplexity, CaoJuan2009, Arun2010, Griffiths2004, Optimum) %>%
  full_join(values_standardized)

# Go from wide to long
values_standardized_Optimum <- values_standardized_Optimum %>% 
  gather(Metric, Value, -Topics, -Split) %>%
  filter(Metric != "Split") %>%
  arrange(Topics, Metric, Split)

# Plot the output
p <- values_standardized_Optimum %>% 
  filter(Metric != "Optimum") %>%
  ggplot(aes_string(x = "Topics", y = "Value", group = "Metric")) + 
  geom_smooth(colour = "red") + 
  geom_point(aes_string(shape = "Metric"), size = 3, alpha = .5) + 
  guides(size = "none", shape = guide_legend(title = "Metrics:")) + 
  scale_x_continuous(breaks = c(seq(from = 0, to = 1500, by = 10))) + 
  labs(x = "Number of Topics", y = NULL) + 
  # facet_grid(Direction ~ .) + 
  theme_classic() %+replace% theme(panel.grid.major.y = element_blank(), 
                                   panel.grid.minor.y = element_blank(), panel.grid.major.x = element_line(colour = "grey70"), 
                                   panel.grid.minor.x = element_blank(), legend.key = element_blank(), 
                                   strip.text.y = element_text(angle = 90))
# Take a look at the plot
p

ggsave(paste0("FIT_NEW_GIBBS_2", ymd(Sys.Date()), "-", hour(Sys.time()), ".", minute(Sys.time()), ".", second(round(Sys.time(), 0)), ".png", sep = ""), width = 20, height = 10, units = "cm")

p2 <- values_standardized_Optimum %>% 
  filter(Metric == "Perplexity") %>%
  ggplot(aes_string(x = "Topics", y = "Value", group = "Metric")) + 
  geom_smooth(colour = "red") + 
  geom_point(aes_string(shape = "Metric"), size = 3, alpha = .5) + 
  guides(size = "none", shape = guide_legend(title = "Metrics:")) + 
  scale_x_continuous(breaks = c(seq(from = 0, to = 1500, by = 10))) + 
  labs(x = "Number of Topics", y = NULL) + 
  # facet_grid(Direction ~ .) + 
  theme_classic() %+replace% theme(panel.grid.major.y = element_blank(), 
                                   panel.grid.minor.y = element_blank(), panel.grid.major.x = element_line(colour = "grey70"), 
                                   panel.grid.minor.x = element_blank(), legend.key = element_blank(), 
                                   strip.text.y = element_text(angle = 90))

p2

# Using Gibbs sampling, let's call it 60 topics 

#  Fit the final LDA model ----
# Create single LDA ----
burnin = 1000
iter = 1000
keep = 50
topics <- 40 # This is what we determined in the previous step
estimation_method <- c("Gibbs") # "Gibbs" or "VEM"

for(i in 1:length(topics)){
  timeStart <- Sys.time()
  alphaPrior <- round(1/topics[i], 3)
  cat("LDA with ", topics[i], " topics, alpha = ", alphaPrior, ", method = ", estimation_method, " starting.  Start time :   ", format(timeStart,'%H:%M:%S'), "\n", sep = "")
  
  if(estimation_method == "Gibbs"){
    
    # Gibbs approach
    itemLDA <- LDA(item_DTM,
                   k = topics[i],
                   method = "Gibbs",
                   control = list(seed = 17,
                                  burnin = burnin,
                                  iter = iter,
                                  keep = keep,
                                  alpha = alphaPrior),
                   mc.cores = 7L,
                   verbose = FALSE)
    
  } else {
    
    # VEM approach
    itemLDA <- LDA(item_DTM,
                   k = topics[i],
                   method = "VEM",
                   control = list(seed = 17,
                                  estimate.alpha = TRUE,
                                  alpha = alphaPrior,
                                  estimate.beta = TRUE),
                   mc.cores = 7L)
    
  }
  
  timeEnd <- Sys.time()
  elapsed <- timeEnd - timeStart
  #if(!dir.exists(paste0(path_output))) dir.create(paste0(path_output), showWarnings = FALSE)
  saveRDS(itemLDA, file = paste0("itemLDA_", estimation_method, "_", topics[i], "K_", alphaPrior, "_Alpha.RData"))
  itemLDA
  cat("LDA with ", topics[i], " topics, alpha = ", alphaPrior, " complete.   Finish time : ",  format(timeEnd,'%H:%M:%S'), "\n", sep = "")
}

#  Initial review of results ----
# The beta matrix tells us how much each term contributes to each topic
itemTopics <- tidy(itemLDA, matrix = "beta")

# Get top 10 terms per topic
# This can provide some insight into what the topics are
item_top_terms <- itemTopics %>%
  arrange(topic, -beta) %>% 
  group_by(topic) %>%
  slice_head(n = 10) %>%
  ungroup() %>% 
  print()

# Plot top terms
item_top_terms  %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 2, scales = "free") +
  coord_flip()
ggsave("topTerms_MATRIX.png", width = 20, height = 80, units = "cm", dpi = 500, limitsize = FALSE)
write.xlsx(item_top_terms, paste0("LDA_item_top_terms.xlsx"))

# The gamma distribution tells us how much each topic contributes to each document/item
# Spread gamma values into a wide, document-by-topic matrix so we can calculate 
# Jensen-Shannon Divergence
itemGamma_wide <- tidy(itemLDA, matrix = "gamma") %>%
  spread(topic, gamma)

write.xlsx(itemGamma_wide, paste0("LDA_itemGamma_wide.xlsx"))

# Calculate Jensen-Shannon Divergence
jensen_shannon_divergence <- CalcJSDivergence(x =  as.matrix(itemGamma_wide[,1:topics[i]+1]), by_rows = TRUE)

# Some of these values can be ever so slightly negative (with Variational Expectation Maximization)
jensen_shannon_divergence[jensen_shannon_divergence < 0]
# So make those zero
jensen_shannon_divergence[jensen_shannon_divergence < 0] <- 0  # There are a few pairs (3-5?) that are infenentesimally negative.  Bump to 0.

# Take ste square root of the Divergence to get Jensen-Shannon Distance (this is the important bit)
jensen_shannon_distance <-as.data.frame(sqrt(jensen_shannon_divergence))

# In the next two steps, pull meaningful labels onto the JSD matrix
jensen_shannon_distance <- jensen_shannon_distance %>% 
  mutate(`Item ID A` = itemGamma_wide$document) %>% 
  select(`Item ID A`, everything())

colnames(jensen_shannon_distance)[2:ncol(jensen_shannon_distance)] <- c(itemGamma_wide$document)

# Pivot longer so we can begin setting up the pairwise data set that 
# we'll pass to the random forest model
itemDistanceLONG <- jensen_shannon_distance %>% 
  pivot_longer(!`Item ID A`, names_to = "Item ID B", values_to = "Distance")

# Now we clean up a little:
# 1) Remove pairs whee Jensen Shannon Distance = 0 (distance from itself)
# 2) Remove second of common comparisons: JSD(Item 1, Item 2) & JSD(Item 2, Item 1)

# Remove same-item pairs 
itemDistanceLONG_Reduced <- itemDistanceLONG %>% 
  filter(`Item ID A` != `Item ID B`) %>% # Remove same-item pairs
  rowwise() %>% 
  mutate(pair_name = paste(sort(c(`Item ID A`, `Item ID B`)), collapse = "-")) %>% 
  distinct(pair_name, .keep_all = TRUE)

# Merge in the Enemy and metadata
ED_df1 <- itemDistanceLONG_Reduced %>%
  rename(Item = `Item ID A`,
         Item2 = `Item ID B`,
         JSD_value = Distance,
         Item_pair = pair_name) %>%
  left_join(enemy_list, by = c("Item" = "Surpass_Id")) %>%
  mutate(Item2_shortID = gsub("^[0-9]*P", "", Item2)) %>%
  mutate(Enemy_Yes = if_else(str_detect(Enemies, Item2_shortID),1,0)) %>%
  mutate(Enemy_Yes = replace_na(Enemy_Yes, 0)) %>%
  distinct(Item_pair, .keep_all = TRUE) %>%
  select(Item, Item2, Item_pair, JSD_value, Enemy_Yes)

ED_df1a <- ED_df1 %>%
  left_join(att, by = c("Item"="Surpass_Id")) %>%
  left_join(att, by = c("Item2"="Surpass_Id")) %>%
  mutate(Same_Item_Type = ifelse(Item_Type.x==Item_Type.y,1,0),
         Same_Content_Outline = ifelse(Content_Outline.x==Content_Outline.y,1,0),
         Same_Has_Exhibit = ifelse(Has_Exhibit.x==Has_Exhibit.y,1,0)) %>%
  select(-Item_Type.x, -Item_Type.y, -Content_Outline.x, -Content_Outline.y,
         -Has_Exhibit.x, -Has_Exhibit.y) %>%
  mutate(Same_Content_Outline = ifelse(is.na(Same_Content_Outline),0, Same_Content_Outline))

ED_df1b <- ED_df1a %>%
  left_join(df, by=c("Item"="Surpass_Id")) %>%
  left_join(df, by=c("Item2"="Surpass_Id")) %>%
  mutate(
    word_count_x = sapply(text.x, word_count),
    word_count_y = sapply(text.y, word_count),
    jaccard_value = mapply(jaccard_value, Keywords.x, Keywords.y)
  ) %>%
  mutate(Avg_Item_Length = rowMeans(cbind(word_count_x, word_count_y), na.rm= TRUE)) %>%
  mutate(Item_ID = as.numeric(gsub("-","", Item_pair)))

ED_df1b <- ED_df1b%>% 
  mutate(Item_ID = as.character(gsub("-","", Item_pair))) %>%
  mutate(Item_ID = as.numeric(gsub("5046P","", Item_ID)))


ED_df1_final <- ED_df1b %>%  select(-Item.x, -Item2, -Item.y, -text.x, -Whole_item.x, -Keywords.x, -Item.y.y,
                                    -text.y, -Whole_item.y, -Keywords.y, -word_count_x, -word_count_y) %>%
  mutate(Item_ID = as.character(gsub("-","", Item_pair))) %>%
  mutate(Item_ID = as.numeric(gsub("5046P","", Item_ID)))	%>%
  select(-Item_pair)

#write.xlsx(ED_df1, paste0(path_output, "LDA_data_output.xlsx"))
# ED_df1 <- read.xlsx("Output/LDA_data_output.xlsx")

#ED_df1_final <- read.xlsx("IPS-Tools/Enemy Detection NLP/Data/SPI_EnemyNLP_final_result.xlsx")

# Exclude rows with missing data
ED_df2_final <- na.omit(ED_df1_final)


###### ML ###################################################################
set.seed(333) # For reproducibility
trainIndex <- createDataPartition(ED_df2_final$Enemy_Yes, p = 0.7, list = FALSE)
train_data <- ED_df2_final[trainIndex,] 
test_data <- ED_df2_final[-trainIndex,] 

table(ED_df2_final$Enemy_Yes)
table(train_data$Enemy_Yes)
table(test_data$Enemy_Yes)

# Apply SMOTE to the training data
smote_result <- SMOTE(train_data, train_data$Enemy_Yes, K = 5, dup_size = 0)
smote_train_data <- smote_result$data
table(smote_train_data$Enemy_Yes)

# Build logistic regression model
glm_model <- glm(Enemy_Yes ~ JSD_value+jaccard_value+Same_Content_Outline+Same_Has_Exhibit+Avg_Item_Length, 
                 data = smote_train_data, 
                 family = binomial)
summary(glm_model)

# Export model results
model_summary <- summary(glm_model)
# Coefficients
coefficients <- model_summary$coefficients[, "Estimate"]
# Standard errors
standard_errors <- model_summary$coefficients[, "Std. Error"]
# P-values
p_values <- model_summary$coefficients[, "Pr(>|z|)"]

result_LR <- data.frame(
  Coefficient = coefficients,
  StdError = standard_errors,
  PValue = p_values
)

write.xlsx(result_LR, "LDA_LR_Summary.xlsx")


# Make predictions on test data
predictions_prob <- predict(glm_model, test_data, type = "response")
predictions_0.7 <- ifelse(predictions_prob >= 0.7, 1, 0)
predictions_0.8 <- ifelse(predictions_prob >= 0.8, 1, 0)
predictions_0.9 <- ifelse(predictions_prob >= 0.9, 1, 0)

# test_data <- test_data %>%
#   mutate(prob_prediction = predictions_prob, predictions_0.7 = predictions_0.7, 
#          predictions_0.8 = predictions_0.8, predictions_0.9 = predictions_0.9)

test_data$prob_prediction <- predictions_prob
test_data$predictions_0.7 <- predictions_0.7
test_data$predictions_0.8 <- predictions_0.8
test_data$predictions_0.9 <- predictions_0.9



# Evaluate
evaluate_model(test_data$Enemy_Yes, predictions_0.7, predictions_prob)
evaluate_model(test_data$Enemy_Yes, predictions_0.8, predictions_prob)
evaluate_model(test_data$Enemy_Yes, predictions_0.9, predictions_prob)


# Apply to the Whole dataset
predictions_prob_all <- predict(glm_model, ED_df2_final, type = "response")
predictions_prob_all <- as.numeric(predictions_prob_all)
ED_df2_final$prob_prediction_LR <- predictions_prob_all




### Random forest ################################################
smote_train_data <- na.omit(smote_train_data)
# Convert to factor if it's categorical
smote_train_data$Enemy_Yes <- as.factor(smote_train_data$Enemy_Yes)

# # Train the random forest model with fewer trees for memory efficiency
# rf_model <- randomForest(Enemy_Yes ~ JSD_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Avg_Item_Length, 
#                          data = smote_train_data, 
#                          ntree = 100, # number of trees. Adjust based on your memory capacity
#                          maxdepth = 10) # maximum depth of the trees. Adjust based on your memory capacity
# print(rf_model)
smote_train_data <- as.data.frame(smote_train_data)
# Train the random forest model with fewer trees for memory efficiency
rf_model <- rfsrc(Enemy_Yes ~ JSD_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Avg_Item_Length, 
                  data = smote_train_data, 
                  ntree = 100, # number of trees
                  nodesize = 10)  # minimum size of terminal nodes

print(rf_model)
summary(rf_model)

# Prediction
predictions_prob_rf <- predict(rf_model, test_data, type = "prob")
prob_class_1 <- predictions_prob_rf$predicted[,"1"]
predictions_rf_0.7 <- ifelse(prob_class_1 >= 0.7, 1, 0)
predictions_rf_0.8 <- ifelse(prob_class_1 >= 0.8, 1, 0)
predictions_rf_0.9 <- ifelse(prob_class_1 >= 0.9, 1, 0)



test_data$prob_prediction <- prob_class_1
test_data$predictions_rf_0.7 <- predictions_rf_0.7
test_data$predictions_rf_0.8 <- predictions_rf_0.8
test_data$predictions_rf_0.9 <- predictions_rf_0.9


# Evaluate the model
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.7, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.8, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.9, prob_class_1)

# Apply to the whole dataset
predictions_prob_all <- predict(rf_model, ED_df2_final, type = "prob")
prob_class_1_all <- predictions_prob_all$predicted[,"1"]

ED_df2_final$prob_prediction_RF <- prob_class_1_all

ED_df2_final <- ED_df2_final %>%
  mutate(enemy_0.7_LR = ifelse(prob_prediction_LR>=0.7, 1, 0),
         enemy_0.8_LR = ifelse(prob_prediction_LR>=0.8, 1, 0),
         enemy_0.9_LR = ifelse(prob_prediction_LR>=0.9, 1, 0),
         enemy_0.7_RF = ifelse(prob_prediction_RF>=0.7, 1, 0),
         enemy_0.8_RF = ifelse(prob_prediction_RF>=0.8, 1, 0),
         enemy_0.9_RF = ifelse(prob_prediction_RF>=0.9, 1, 0))


write.xlsx(ED_df2_final, "LDA_output_data.xlsx")

# Summarize results
# Number of predicted enemy items at different probabilities
ED_df2_final %>% filter(prob_prediction_LR >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))
ED_df2_final %>% filter(prob_prediction_LR >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))
ED_df2_final %>% filter(prob_prediction_LR >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))

ED_df2_final %>% filter(prob_prediction_RF >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))
ED_df2_final %>% filter(prob_prediction_RF >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))
ED_df2_final %>% filter(prob_prediction_RF >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df2_final))


# DESCRIPTIVE STATISTICS OF COSINE SIMILARITY INDICES FROM THE LDA
LDA_desc_stats <- ED_df2_final %>% group_by(Enemy_Yes) %>% summarise(
  across(everything(), list(
    count = ~n(),
    mean = ~mean(.),
    sd = ~sd(.),
    min_value = ~min(.),
    Q1 = ~quantile(., 0.25),
    median = ~median(.),
    Q3 = ~quantile(., 0.75),
    max_value = ~max(.)
  ))
)

write.xlsx(LDA_desc_stats, "LDA_desc_stats.xlsx")

# Add the topic
# Find the highest, 2nd highest, and 3rd highest values for each row, along with their corresponding column headers.
# Read the data
LDA_topic <- read.xlsx("LDA_itemGamma_wide.xlsx")

# Function to get the top 3 column names
get_top_3_names <- function(row) {
  order(row, decreasing = TRUE)[1:3]
}

# Function to get the top 3 values
get_top_3_values <- function(row) {
  sort(row, decreasing = TRUE)[1:3]
}

# Apply the functions across the rows (excluding the 'document' column)
top_3_names <- apply(LDA_topic[,-1], 1, get_top_3_names)
top_3_values <- apply(LDA_topic[,-1], 1, get_top_3_values)

# Add the columns to the original data
LDA_topic$highest_column <- top_3_names[1,]
LDA_topic$second_highest_column <- top_3_names[2,]
LDA_topic$third_highest_column <- top_3_names[3,]

LDA_topic$highest_value <- top_3_values[1,]
LDA_topic$second_highest_value <- top_3_values[2,]
LDA_topic$third_highest_value <- top_3_values[3,]

# Keep topics that have probability >=0.3
LDA_topic_final <- LDA_topic %>% mutate(highest_column = ifelse(highest_value >=0.3, highest_column, NA),
                                        second_highest_column = ifelse(second_highest_value >=0.3, second_highest_column, NA),
                                        third_highest_column = ifelse(third_highest_value >=0.3, third_highest_column, NA)) %>% 
  select(document, highest_column, second_highest_column, third_highest_column) %>%
  rename(Item = document, topic_1 = highest_column, topic_2 = second_highest_column, topic_3 = third_highest_column)

####
LDA_topic_words <- read.xlsx("LDA_item_top_terms.xlsx")

# Function to concatenate terms
concatenate_terms <- function(terms) {
  paste(terms, collapse = " ")
}
# Find the top terms for each topic and concatenate them to form a theme
LDA_topic_themes <- LDA_topic_words %>%
  group_by(topic) %>%
  arrange(desc(beta)) %>%
  slice_head(n = 3) %>%
  summarise(theme = paste(term, collapse = " "), .groups = 'drop')

LDA_topic_final <- LDA_topic_final %>%
  left_join(LDA_topic_themes, by = c("topic_1" = "topic")) %>%
  left_join(LDA_topic_themes, by = c("topic_2" = "topic")) %>%
  left_join(LDA_topic_themes, by = c("topic_3" = "topic")) %>%
  mutate(topic = paste(theme.x, theme.y, theme)) %>%
  mutate(topic = gsub(" NA NA", "", topic)) %>%
  mutate(topic = gsub(" NA", "", topic))

write.xlsx(LDA_topic_final, "LDA_topic_final.xlsx")

################### Rerun the model after updating the enemy column
set.seed(3333) # For reproducibility

# Run this after updating the enemy item label in the first round
ED_df2_final <- LDA_output_data2 %>%
  mutate(Same_Topic = ifelse(topic.x == topic.y, 1, 0)) %>%
  select(Item_ID, Enemy_Yes, Same_Content_Outline, Same_Has_Exhibit, jaccard_value, Same_Topic,
         Avg_Item_Length, JSD_value) 

trainIndex <- createDataPartition(ED_df2_final$Enemy_Yes, p = 0.7, list = FALSE)
train_data <- ED_df2_final[trainIndex,] 
test_data <- ED_df2_final[-trainIndex,] 

table(ED_df2_final$Enemy_Yes)
table(train_data$Enemy_Yes)
table(test_data$Enemy_Yes)

# Apply SMOTE to the training data
smote_result <- SMOTE(train_data, train_data$Enemy_Yes, K = 5, dup_size = 0)
smote_train_data <- smote_result$data
table(smote_train_data$Enemy_Yes)

# Convert to factor if it's categorical
smote_train_data <- na.omit(smote_train_data)
smote_train_data$Enemy_Yes <- as.factor(smote_train_data$Enemy_Yes)
smote_train_data <- as.data.frame(smote_train_data)


# Tune hyperparamters
# Set up train control with repeated cross-validation
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     search = "grid")


# Define the hyperparameter grid
grid <- expand.grid(.nodesize = c(5, 10, 15, 20),
                    .ntree = c(50, 100, 200))

# Initialize variables to store the best model's hyperparameters and accuracy
best_nodesize <- NULL
best_ntree <- NULL
best_auc <- 0

# Iterate over the hyperparameter grid
for(i in 1:nrow(grid)) {
  set.seed(123)  # for reproducibility
  
  # Train the random forest model
  rf <- rfsrc(Enemy_Yes ~ JSD_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Same_Topic + Avg_Item_Length, 
              data = smote_train_data, 
              nodesize = grid$.nodesize[i],
              ntree = grid$.ntree[i])  # minimum size of terminal nodes
  
  # Predict probabilities for the validation set
  pred_prob <- predict(rf, newdata = test_data, type = 'prob')$predicted
  
  # Compute AUC for the model
  roc_obj <- roc(test_data$Enemy_Yes, pred_prob[,2])  # Assuming 2nd column has probability for the positive class
  auc_val <- auc(roc_obj)
  
  # Update best hyperparameters and AUC if current model is better
  if(auc_val > best_auc) {
    best_auc <- auc_val
    best_nodesize <- grid$.nodesize[i]
    best_ntree <- grid$.ntree[i]
  }
}

# Print the best hyperparameters and accuracy
print(paste(
            "Best nodesize:", best_nodesize, 
            "Best ntree:", best_ntree, 
            "with AUC:", best_auc))

# Run the improvised model
# Train the random forest model with fewer trees for memory efficiency
rf_model <- rfsrc(Enemy_Yes ~ JSD_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Same_Topic + Avg_Item_Length, 
                  data = smote_train_data, 
                  ntree = 200, # number of trees
                  nodesize = 10)  # minimum size of terminal nodes

print(rf_model)
summary(rf_model)

# Prediction
predictions_prob_rf <- predict(rf_model, test_data, type = "prob")
prob_class_1 <- predictions_prob_rf$predicted[,"1"]
predictions_rf_0.7 <- ifelse(prob_class_1 >= 0.7, 1, 0)
predictions_rf_0.8 <- ifelse(prob_class_1 >= 0.8, 1, 0)
predictions_rf_0.9 <- ifelse(prob_class_1 >= 0.9, 1, 0)



test_data$prob_prediction <- prob_class_1
test_data$predictions_rf_0.7 <- predictions_rf_0.7
test_data$predictions_rf_0.8 <- predictions_rf_0.8
test_data$predictions_rf_0.9 <- predictions_rf_0.9


# Evaluate the model
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.7, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.8, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.9, prob_class_1)

# Apply to the whole dataset
predictions_prob_all <- predict(rf_model, ED_df2_final, type = "prob")
prob_class_1_all <- predictions_prob_all$predicted[,"1"]

ED_df2_final$prob_prediction_RF <- prob_class_1_all

ED_df2_final <- ED_df2_final %>%
  mutate(enemy_0.7_RF = ifelse(prob_prediction_RF>=0.7, 1, 0),
         enemy_0.8_RF = ifelse(prob_prediction_RF>=0.8, 1, 0),
         enemy_0.9_RF = ifelse(prob_prediction_RF>=0.9, 1, 0))


write.xlsx(ED_df2_final, "LDA_output_data_new.xlsx")

