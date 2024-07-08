# Find the frequency of words to add to stopwords
# Create a text corpus
corpus <- Corpus(VectorSource(bank_clean$text))

stoplist <- stop_words

# Text preprocessing
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lower case
corpus <- tm_map(corpus, removePunctuation) # Remove punctuation
corpus <- tm_map(corpus, removeNumbers) # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en")) # Remove English common stopwords
corpus <- tm_map(corpus, stripWhitespace) # Remove whitespace
corpus <- tm_map(corpus, lemmatize_strings) # Lemmatize words

# Create a Term Document Matrix
tdm <- TermDocumentMatrix(corpus)

# Convert the term-document matrix into a data frame
tdm_df <- as.data.frame(as.matrix(tdm))

# Sum up the occurrences of each word
word_freqs <- row_sums(tdm_df)

# Create a data frame of words and their frequencies
word_freqs_df <- data.frame(word = names(word_freqs), freq = word_freqs)

# Sort the data frame by frequency in descending order
word_freqs_df <- word_freqs_df[order(-word_freqs_df$freq),]

# Calculate the prob
word_freqs_df$prob <- word_freqs_df$freq/sum(word_freqs_df$freq)

# Select words that have probability > 0.01 and add to remove_words
selected_stop_words <- word_freqs_df %>% filter(prob > 0.01) %>% select(word) %>% as.list()

# Update the remove_words list and clean the text again
remove_words <- c(remove_words, selected_stop_words$word)

bank_clean <- clean_text(bank_clean, "text", remove_words)

#####################################
# Extracting N-Grams and Collocations
#####################################

# Clean Stem
#clean corpus
stem_clean <- bank$Stem %>%
  stringr::str_to_title()
# tokenize corpus
stem_tokzd <- quanteda::tokens(stem_clean)
# extract bigrams
stem_BiGrams <- stem_tokzd %>% 
  quanteda::tokens_remove(stopwords("en")) %>% 
  quanteda::tokens_select(pattern = "^[A-Z]", 
                          valuetype = "regex",
                          case_insensitive = FALSE, 
                          padding = TRUE) %>% 
  quanteda.textstats::textstat_collocations(min_count = 5, tolower = FALSE)

stem_ngram_extract <- quanteda::tokens_compound(stem_tokzd, pattern = stem_BiGrams)

# Convert tokens object to dfm
stem_dfm_object <- quanteda::dfm(stem_ngram_extract)
# Convert dfm to data.frame
stem_df <- quanteda::convert(stem_dfm_object, to = "data.frame")
stem_df_long <- tidyr::gather(stem_df, key = "variable", value = "value") %>% distinct(variable)
stem_df_filtered <- stem_df_long %>% 
  filter(grepl("_", variable)) %>%
  rename(collocate = variable) %>%
  mutate(phrase = gsub("_", " ", collocate))


# Clean key_answer
#clean corpus
key_clean <- bank$key_answer %>%
  stringr::str_to_title()
# tokenize corpus
key_tokzd <- quanteda::tokens(key_clean)
# extract bigrams
key_BiGrams <- key_tokzd %>% 
  quanteda::tokens_remove(stopwords("en")) %>% 
  quanteda::tokens_select(pattern = "^[A-Z]", 
                          valuetype = "regex",
                          case_insensitive = FALSE, 
                          padding = TRUE) %>% 
  quanteda.textstats::textstat_collocations(min_count = 5, tolower = FALSE)

key_ngram_extract <- quanteda::tokens_compound(key_tokzd, pattern = key_BiGrams)

# Convert tokens object to dfm
key_dfm_object <- quanteda::dfm(key_ngram_extract)
# Convert dfm to data.frame
key_df <- quanteda::convert(key_dfm_object, to = "data.frame")
key_df_long <- tidyr::gather(key_df, key = "variable", value = "value") %>% distinct(variable)
key_df_filtered <- key_df_long %>% 
  filter(grepl("_", variable)) %>%
  rename(collocate = variable) %>%
  mutate(phrase = gsub("_", " ", collocate))

# Combine collocate lists from stem and key
collocate_df <- rbind(stem_df_filtered, key_df_filtered) %>% distinct()


# Clean the text column and find collocate words
bank_clean <- bank_clean %>%
  mutate(text_clean = sapply(text, combine_phrases, collocation_list = collocate_df$phrase))

# Separate the collocates out. We will combine them to the keywords later
bank_clean <- bank_clean %>%
  mutate(collocate = str_extract_all(text_clean, "\\b\\w+\\_\\w+\\b") %>% 
           sapply(function(x) paste(x, collapse = "/"))) %>%
  mutate(text_clean = str_replace_all(text_clean, "\\b\\w+\\_\\w+\\b", ""))

# Apply tf_idf function to find new keywords
new_keywords_tfidf_df <- get_keywords_tfidf(bank_clean, "text_clean", "Item") %>% rename(new_keywords_tfidf = keywords_tfidf)

# Merge into the bank dataset
bank_clean2 <- bank_clean %>% 
  left_join(new_keywords_tfidf_df, by = c("Item" = "doc_id")) %>%
  mutate(new_keywords_tfidf = paste0(new_keywords_tfidf, " ", collocate, " ", Item_Name)) %>%
  mutate(new_keywords_tfidf = gsub("_", " ", new_keywords_tfidf)) %>%
  mutate(new_keywords_tfidf = sapply(new_keywords_tfidf, remove_duplicate_words)) %>%
  mutate(new_keywords_tfidf = str_replace_all(new_keywords_tfidf, "NA", ""))

# Rerun another cleaning and TFIDF for keywords
bank_clean2 <- clean_text(bank_clean2, "new_keywords_tfidf", remove_words)
bank_clean2[["new_keywords_tfidf"]] <- gsub("\\d", "", bank_clean2[["new_keywords_tfidf"]])

new_keywords_tfidf_df2 <- get_keywords_tfidf(bank_clean2, "new_keywords_tfidf", "Item") %>% rename(new_keywords_tfidf = keywords_tfidf)

bank_clean2 <- bank_clean %>% 
  left_join(new_keywords_tfidf_df2, by = c("Item" = "doc_id")) %>%
  mutate(new_keywords_tfidf = ifelse(is.na(new_keywords_tfidf), Item_Name, new_keywords_tfidf)) %>%
  filter(new_keywords_tfidf!="")
