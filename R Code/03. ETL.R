
# Load in the data from Surpass (for content) 
Surpass_bank <- api.getBank(exam = "VT", itemContent = TRUE)
#x <- Surpass_bank[,"Surpass_Id"]
#Surpass_bank_content <- oapi.getItems(x, itemContent = TRUE)
write.csv(Surpass_bank, "VT_Surpass_bank_content.csv", row.names = F)

# Combine the Stem and key_answer
Surpass_bank_content <- read.csv("Data/VT/VT_Surpass_bank_content.csv") 
Surpass_bank_content <- Surpass_bank
Surpass_bank_content %>% group_by(Item_Status) %>% count()
Surpass_bank_content %>% group_by(Item_Type) %>% count()

bank <- Surpass_bank_content %>% select(Surpass_Id, Item, Exam, Item_Status, Item_Type, Key, Enemies, Enemies_Name,Has_Exhibit, Media_Item_Type,
                                        Domain, Content_Outline, Stem, Distractor_A, Distractor_B, Distractor_C, Distractor_D, Item_Name) %>%
  mutate(key_answer = case_when(Key=="A"~Distractor_A,
                                Key=="B"~Distractor_B,
                                Key=="C"~Distractor_C,
                                Key=="D"~Distractor_D)) %>% 
  filter(Item_Status=="Operational" |
           Item_Status=="Pretest ready" |
           Item_Status=="Pretest") %>%
  filter(Item_Type=="SSMC") %>%
  mutate(Whole_item = paste0(Stem, "/ ", Distractor_A, "/ ", Distractor_B, "/ ", 
                             Distractor_C, "/ ", Distractor_D)) %>%
  mutate(text = as.character(paste(Stem, key_answer)), Item = as.character(Item)) %>%
  select(Surpass_Id, Item, Stem, key_answer, Whole_item, text, Item_Name) %>%
  filter(key_answer!="")


# Clean the text and keywords columns
bank_clean <- clean_text(bank, "Stem", remove_words)
bank_clean <- clean_text(bank_clean, "key_answer", remove_words)
bank_clean <- clean_text(bank_clean, "Whole_item", remove_words)
bank_clean <- clean_text(bank_clean, "text", remove_words)
bank_clean <- clean_text(bank_clean, "Item_Name", remove_words)

att <- Surpass_bank_content %>% select(Surpass_Id, Item_Type, Content_Outline, Has_Exhibit) %>%
  mutate(Has_Exhibit = as.numeric(Has_Exhibit)) %>%
  mutate(Item_Type = as.factor(Item_Type),
         Content_Outline = as.factor(Content_Outline)) %>%
  mutate(Surpass_Id = as.character(Surpass_Id))
