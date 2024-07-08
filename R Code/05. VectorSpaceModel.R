# Vector Space Model
set.seed(11)
# Load in the data
df <- bank_clean2 %>%
  select(Surpass_Id, Item, text, Whole_item,new_keywords_tfidf) %>%
  rename(Keywords = new_keywords_tfidf)%>%
  mutate(Surpass_Id=as.character(Surpass_Id)) 
  #slice_sample(n = 500, replace = FALSE)
# stop_words object combines the SMART, Snowball, and Onix stop word sets
stoplist <- stop_words
enemy_list <- Surpass_bank %>% select(Surpass_Id, Enemies) %>% 
  mutate(Surpass_Id = as.character(Surpass_Id)) %>%
  filter(!is.na(Enemies))

# Analysis 1: Stem-key 
# Extract the stem and clean again
stem<-removePunctuation(df$text)
stem<-tolower(stem)
stem<-removeNumbers(stem)
stem<-str_replace(stem,"\t"," ")
stem<-str_replace(stem,"\n"," ")
stem<-stripWhitespace(stem)

#Create a matrix with a column for item id and a column for each word in the
#item component: lemmatizing and stemming each word here. 
idword<-matrix("",ncol=4)
colnames(idword)<-c("ID","word","stem","lemma")
idword<-idword[-1,]

for(i in 1:length(stem)){
  sp<-stem[i]
  sp<-str_split(sp," ")
  #stem words 
  sp2<-stemDocument(stem[i])
  sp2<-str_split(sp2," ")
  #lemmatize words
  sp3<-lemmatize_strings(stem[i])
  sp3<-str_split(sp3," ")
  idword<-rbind(idword,cbind(rep(paste("stem_",df$Surpass_Id[i],sep=""),length(sp2[[1]])),sp[[1]],sp2[[1]],sp3[[1]]))
}

idword<-subset(idword,idword[,2]!="")

stop_words <- stoplist[, 1]
idword <- idword[!idword[, 2] %in% stop_words, ]

#Create a document term matrix
dwi<-table(idword[,1],idword[,3])
dwicos<-cosine(t(dwi))
stempairs<-data.frame(rep(rownames(dwicos),times=ncol(dwicos)),stack(data.frame(dwicos)))

stempairs<-subset(stempairs,stempairs[,1]!=stempairs[,3])
colnames(stempairs)[colnames(stempairs)==colnames(stempairs)[1]] <- "Group1"
colnames(stempairs)[colnames(stempairs)==colnames(stempairs)[3]] <- "Group2"

stempairs$Item <- gsub("[^0-9]", "", stempairs$Group1)
stempairs$Item2 <- gsub("[^0-9]", "", stempairs$Group2)

ED_df1 <- stempairs %>%
  left_join(enemy_list, by = c("Item" = "Surpass_Id")) %>%
  mutate(Enemy_Yes = if_else(str_detect(Enemies, Item2),1,0)) %>%
  mutate(Enemy_Yes = replace_na(Enemy_Yes, 0)) %>%
  mutate(Item_pair = if_else(as.numeric(Item)<as.numeric(Item2), paste0(Item, "-", Item2), paste0(Item2, "-", Item))) %>%
  distinct(Item_pair, .keep_all = TRUE) %>%
  select(Item, Item2, Item_pair, values, Enemy_Yes)

ED_df1 <- ED_df1 %>%
  left_join(att, by = c("Item"="Surpass_Id")) %>%
  left_join(att, by = c("Item2"="Surpass_Id")) %>%
  rename(Cosine_value = values) %>%
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
set.seed(111) # For reproducibility
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

write.xlsx(result_LR, "VSM_LR_Summary.xlsx")


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
# rf_model <- randomForest(Enemy_Yes ~ Cosine_value + jaccard_value + Same_Domain + Same_Subdomain + Same_Task + Same_Has_Exhibit + Avg_Item_Length, 
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

# test_data$prob_prediction <- prob_class_1
# test_data$predictions_rf_0.7 <- predictions_rf_0.7
# test_data$predictions_rf_0.8 <- predictions_rf_0.8
# test_data$predictions_rf_0.9 <- predictions_rf_0.9

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


write.xlsx(ED_df1_final, "VSM_output_data.xlsx")


# Summarize results
# Number of predicted enemy items at different probabilities
ED_df1_final %>% filter(prob_prediction_LR >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_LR >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_LR >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))

ED_df1_final %>% filter(prob_prediction_RF >= 0.7) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_RF >= 0.8) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))
ED_df1_final %>% filter(prob_prediction_RF >= 0.9) %>% summarize(N = n(), Perc = N/nrow(ED_df1_final))


# DESCRIPTIVE STATISTICS OF COSINE SIMILARITY INDICES FROM THE VSM
VSM_desc_stats <- ED_df1_final %>% group_by(Enemy_Yes) %>% summarise(
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

write.xlsx(VSM_desc_stats, "VSM_desc_stats.xlsx")
