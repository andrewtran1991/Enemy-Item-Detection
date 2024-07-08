set.seed(22)
# Load in the data
df <- bank_clean2 %>%
  select(Surpass_Id, Item, text, Whole_item,new_keywords_tfidf) %>%
  rename(Keywords = new_keywords_tfidf)%>%
  mutate(Surpass_Id=as.character(Surpass_Id)) 
df <- na.omit(df)

enemy_list <- Surpass_bank %>% select(Surpass_Id, Enemies) %>% 
  mutate(Surpass_Id = as.character(Surpass_Id)) %>%
  filter(!is.na(Enemies))

dtm <- CreateDtm(doc_vec = df$text, # character vector of documents
                 doc_names = df$Surpass_Id, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus = 2) # default is all available cpus on the system

dtm <- dtm[,colSums(dtm) > 2]


# get a tf-idf matrix
tf_sample <- TermDocFreq(dtm)

tf_sample$idf[ is.infinite(tf_sample$idf) ] <- 0 # fix idf for missing words

tf_idf <- t(dtm / rowSums(dtm)) * tf_sample$idf

tf_idf <- t(tf_idf)

min_term_freq <- 10
doc_freqs <- rowSums(tf_idf>0)

doc_freqs[is.na(doc_freqs)] <- 0
tf_idf <- tf_idf[doc_freqs > min_term_freq, ]

# Fit a Latent Semantic Analysis model
# Use the number of topic from LDA
lsa_model <- FitLsaModel(dtm = tf_idf, k = 50)

str(lsa_model)
summary(lsa_model$coherence)
hist(lsa_model$coherence, col= "blue")

# Get the top terms of each topic
lsa_model$top_terms <- GetTopTerms(phi = lsa_model$phi, M = 5)
head(t(lsa_model$top_terms))

# Get the prevalence of each topic
# Applying a threshold 0.05, for topics in/out of docuemnts. 
lsa_model$prevalence <- colSums(lsa_model$theta) / sum(lsa_model$theta) * 100

# textmineR has a naive topic labeling tool based on probable bigrams
lsa_model$labels <- LabelTopics(assignments = lsa_model$theta > 0.05, 
                                dtm = dtm,
                                M = 1)

# put them together, with coherence into a summary table
lsa_model$summary <- data.frame(topic = rownames(lsa_model$phi),
                                label = lsa_model$labels,
                                coherence = round(lsa_model$coherence, 3),
                                prevalence = round(lsa_model$prevalence,3),
                                top_terms = apply(lsa_model$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)
lsa_model$summary[ order(lsa_model$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ]

# set up the assignments matrix and a simple dot product gives us predictions
lsa_assignments <- t(dtm / rowSums(dtm)) * tf_sample$idf

lsa_assignments <- t(lsa_assignments)

lsa_assignments <- predict(lsa_model, lsa_assignments)

# Cosine similarity matrix
cosine_mat <- as.matrix(proxy::simil(x = lsa_assignments, method = "cosine"))
cat('shape of cosine matrix:', dim(cosine_mat), '\n')
as.data.frame(cosine_mat)
stempairs <- melt(cosine_mat)

stempairs<-subset(stempairs,stempairs[,1]!=stempairs[,2])
colnames(stempairs)[colnames(stempairs)==colnames(stempairs)[1]] <- "Group1"
colnames(stempairs)[colnames(stempairs)==colnames(stempairs)[2]] <- "Group2"

stempairs$Item <- gsub("[^0-9]", "", stempairs$Group1)
stempairs$Item2 <- gsub("[^0-9]", "", stempairs$Group2)
stempairs <- stempairs %>%
  rename(Cosine_value = value)

ED_df1 <- stempairs %>%
  left_join(enemy_list, by = c("Item" = "Surpass_Id")) %>%
  mutate(Enemy_Yes = if_else(str_detect(Enemies, Item2),1,0)) %>%
  mutate(Enemy_Yes = replace_na(Enemy_Yes, 0)) %>%
  mutate(Item_pair = if_else(as.numeric(Item)<as.numeric(Item2), paste0(Item, "-", Item2), paste0(Item2, "-", Item))) %>%
  distinct(Item_pair, .keep_all = TRUE) %>%
  select(Item, Item2, Item_pair, Cosine_value, Enemy_Yes)

ED_df1 <- ED_df1 %>%
  left_join(att, by = c("Item"="Surpass_Id")) %>%
  left_join(att, by = c("Item2"="Surpass_Id")) %>%
  mutate(Same_Item_Type = ifelse(Item_Type.x==Item_Type.y,1,0),
         Same_Content_Outline = ifelse(Content_Outline.x==Content_Outline.y,1,0),
         Same_Has_Exhibit = ifelse(Has_Exhibit.x==Has_Exhibit.y,1,0)) %>%
  select(-Item_Type.x, -Item_Type.y, -Content_Outline.x, -Content_Outline.y,
         -Has_Exhibit.x, -Has_Exhibit.y)

ED_df1 <- ED_df1 %>%
  left_join(df, by=c("Item"="Surpass_Id")) %>%
  left_join(df, by=c("Item2"="Surpass_Id")) %>%
  mutate(
    word_count_x = sapply(text.x, word_count),
    word_count_y = sapply(text.y, word_count),
    jaccard_value = mapply(jaccard_value, Keywords.x, Keywords.y)
  ) %>%
  mutate(Avg_Item_Length = rowMeans(cbind(word_count_x, word_count_y), na.rm= TRUE)) %>%
  mutate(Item_ID = as.numeric(gsub("-","", Item_pair)))

ED_df1_final <- ED_df1 %>%  select(-Item_pair, -Item.x, -Item2, -Item.y, -text.x, -Whole_item.x, -Keywords.x, -Item.y.y,
                                   -text.y, -Whole_item.y, -Keywords.y, -word_count_x, -word_count_y) 

# Exclude rows with missing data
ED_df1_final <- na.omit(ED_df1_final)
###### ML ############
set.seed(222) # For reproducibility
trainIndex <- createDataPartition(ED_df1_final$Enemy_Yes, p = 0.7, list = FALSE)
train_data <- ED_df1_final[trainIndex,] 
test_data <- ED_df1_final[-trainIndex,] 

table(train_data$Enemy_Yes)
table(ED_df1_final$Enemy_Yes)

# Apply SMOTE to the training data
smote_result <- SMOTE(train_data, train_data$Enemy_Yes, K = 5, dup_size = 0)
smote_train_data <- smote_result$data
table(smote_train_data$Enemy_Yes)

# Build logistic regression model
glm_model <- glm(Enemy_Yes ~ Cosine_value+jaccard_value+Same_Content_Outline+Same_Has_Exhibit+Avg_Item_Length, 
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

write.xlsx(result_LR, "LSA_LR_Summary.xlsx")


# Make predictions on test data
predictions_prob <- predict(glm_model, test_data, type = "response")
predictions_0.7 <- ifelse(predictions_prob >= 0.7, 1, 0)
predictions_0.8 <- ifelse(predictions_prob >= 0.8, 1, 0)
predictions_0.9 <- ifelse(predictions_prob >= 0.9, 1, 0)

test_data <- test_data %>%
  mutate(prob_prediction = predictions_prob, predictions_0.7 = predictions_0.7, 
         predictions_0.8 = predictions_0.8, predictions_0.9 = predictions_0.9)

# Evaluate
evaluate_model(test_data$Enemy_Yes, predictions_0.7, predictions_prob)
evaluate_model(test_data$Enemy_Yes, predictions_0.8, predictions_prob)
evaluate_model(test_data$Enemy_Yes, predictions_0.9, predictions_prob)


# Apply to the Whole dataset
predictions_prob_all <- predict(glm_model, ED_df1_final, type = "response")
ED_df1_final <- ED_df1_final %>%
  mutate(prob_prediction_LR = predictions_prob_all)



### Random forest ################################################
smote_train_data <- na.omit(smote_train_data)
# Convert to factor if it's categorical
smote_train_data$Enemy_Yes <- as.factor(smote_train_data$Enemy_Yes)

# # Train the random forest model with fewer trees for memory efficiency
# rf_model <- randomForest(Enemy_Yes ~ Cosine_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Avg_Item_Length, 
#                          data = smote_train_data, 
#                          ntree = 100, # number of trees. Adjust based on your memory capacity
#                          maxdepth = 10) # maximum depth of the trees. Adjust based on your memory capacity
# print(rf_model)

# Train the random forest model with fewer trees for memory efficiency
rf_model <- rfsrc(Enemy_Yes ~ Cosine_value + jaccard_value + Same_Content_Outline + Same_Has_Exhibit + Avg_Item_Length, 
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

test_data <- test_data %>%
  mutate(prob_prediction = prob_class_1, predictions_rf_0.7 = predictions_rf_0.7,
         predictions_rf_0.8 = predictions_rf_0.8, predictions_rf_0.9 = predictions_rf_0.9)

# Evaluate the model
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.7, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.8, prob_class_1)
evaluate_model(test_data$Enemy_Yes, predictions_rf_0.9, prob_class_1)

# Apply to the whole dataset
predictions_prob_all <- predict(rf_model, ED_df1_final, type = "prob")
prob_class_1_all <- predictions_prob_all$predicted[,"1"]

ED_df1_final <- ED_df1_final %>%
  mutate(prob_prediction_RF = prob_class_1_all) %>%
  mutate(enemy_0.7_LR = ifelse(prob_prediction_LR>=0.7, 1, 0),
         enemy_0.8_LR = ifelse(prob_prediction_LR>=0.8, 1, 0),
         enemy_0.9_LR = ifelse(prob_prediction_LR>=0.9, 1, 0),
         enemy_0.7_RF = ifelse(prob_prediction_RF>=0.7, 1, 0),
         enemy_0.8_RF = ifelse(prob_prediction_RF>=0.8, 1, 0),
         enemy_0.9_RF = ifelse(prob_prediction_RF>=0.9, 1, 0))


write.xlsx(ED_df1_final, "LSA_output_data.xlsx")


# Summarize results
# Number of predicted enemy items at different probabilities
ED_df1_final %>% filter(prob_prediction_LR >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_LR >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_LR >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))

ED_df1_final %>% filter(prob_prediction_RF >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_RF >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_RF >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))


# DESCRIPTIVE STATISTICS OF COSINE SIMILARITY INDICES FROM THE LSA
LSA_desc_stats <- ED_df1_final %>% group_by(Enemy_Yes) %>% summarise(
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

write.xlsx(LSA_desc_stats, "LSA_desc_stats.xlsx")


