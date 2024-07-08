# Load in the LDA result file
LDA_output_data <- read.xlsx("LDA_output_data.xlsx")
LDA_output_data %>% filter(Enemy_Yes==1) %>% summarize(N = n(), Perc = N/nrow(LDA_output_data))
LDA_output_data %>% filter(prob_prediction_RF >= 0.7) %>% summarize(N = n(), Perc = N/nrow(LDA_output_data))

# Merge in the Item ID and its original content
ID_data <- ED_df1 %>% select(Item_ID, Item.x, Item2) %>% rename(Item.y = Item2)
Content_data <- bank %>% select(Surpass_Id, text) %>% mutate(Surpass_Id = as.character(Surpass_Id))
Topic_data <- LDA_topic_final %>% select(Item, topic)

LDA_output_data2 <- LDA_output_data %>% 
  left_join(ID_data, by = "Item_ID") %>%
  left_join(Content_data, by = c("Item.x" = "Surpass_Id")) %>%
  left_join(Topic_data, by = c("Item.x" = "Item")) %>%
  left_join(Content_data, by = c("Item.y" = "Surpass_Id")) %>%
  left_join(Topic_data, by = c("Item.y" = "Item")) 
  

# Update the enemy column using enemy_0.7_RF
LDA_output_data2 <- LDA_output_data2 %>% 
  mutate(Enemy_Original = Enemy_Yes) %>%
  mutate(Enemy_Yes = ifelse(enemy_0.7_RF==1 | Enemy_Original==1, 1, 0 )) 
  
LDA_output_data2 %>% filter(Enemy_Yes==1) %>% summarize(N = n(), Perc = N/nrow(LDA_output_data))

write.xlsx(LDA_output_data2, "LDA_output_data2.xlsx")

########################## After rerunning the model
# Load in the LDA result file
Enemy_original_data <- LDA_output_data2 %>% select(Item_ID, Enemy_Original)

LDA_output_data_new <- read.xlsx("LDA_output_data_new.xlsx")
LDA_output_data3 <- LDA_output_data_new %>% 
  left_join(ID_data, by = "Item_ID") %>%
  left_join(Content_data, by = c("Item.x" = "Surpass_Id")) %>%
  left_join(Topic_data, by = c("Item.x" = "Item")) %>%
  left_join(Content_data, by = c("Item.y" = "Surpass_Id")) %>%
  left_join(Topic_data, by = c("Item.y" = "Item")) %>%
  left_join(Enemy_original_data, by = "Item_ID")
  

LDA_output_data3 %>% filter(Enemy_Yes==1) %>% summarize(N = n(), Perc = N/nrow(LDA_output_data))
LDA_output_data3 %>% filter(prob_prediction_RF >= 0.7) %>% summarize(N = n(), Perc = N/nrow(LDA_output_data))

write.xlsx(LDA_output_data3, "LDA_output_data_final.xlsx")
