# Download the English model for udpipe
ud_model <- udpipe_download_model(language = "english")
# Load the English model
ud_model <- udpipe_load_model(ud_model$file_model)

# List of words to be removed from the sentence
remove_words = c("image", "images", "video", "videos", "findings", "finding", "send", "commonly", "common", "present", "demonstrate", "demonstrated", "likely", "best")

# Function to clean the text
clean_text <- function(df, column, remove_words) {
  # Remove non-ASCII characters
  df[[column]] <- iconv(df[[column]], "UTF-8", "ASCII", sub = " ")
  
  # Make the text lowercase
  df[[column]] <- tolower(df[[column]])
  
  # Trim spaces from the front and back
  df[[column]] <- trimws(df[[column]])
  
  # Remove non-alphanumeric characters
  df[[column]] <- gsub("[^[:alnum:] ]", " ", df[[column]])
  
  # Remove numbers
  df[[column]] <- gsub("\\d", "", df[[column]])
  
  # Remove the specified words
  for (remove_word in remove_words) {
    df[[column]] <- gsub(paste0("\\b", remove_word, "\\b"), "", df[[column]])
  }
  
  # Return the cleaned dataframe
  return(df)
}


# Function to clean the text
clean_text_blob <- function(text, remove_words) {
  # Remove non-ASCII characters
  text <- iconv(text, "UTF-8", "ASCII", sub = " ")
  
  # Make the text lowercase
  text <- tolower(text)
  
  # Trim spaces from the front and back
  text <- trimws(text)
  
  # Remove non-alphanumeric characters
  text <- gsub("[^[:alnum:] ]", " ", text)
  
  # Remove numbers
  text <- gsub("\\d", "", text)
  
  # Remove the specified words
  for (remove_word in remove_words) {
    text <- gsub(paste0("\\b", remove_word, "\\b"), "", text)
  }
  
  # Return the cleaned dataframe
  return(text)
}


# Function to remove duplicate words from a sentence
remove_duplicate_words <- function(sentence) {
  # Split the sentence into words
  words <- unlist(strsplit(sentence, "\\s+"))
  
  # Remove duplicates
  unique_words <- unique(words)
  
  # Collapse the words back into a sentence
  unique_sentence <- paste(unique_words, collapse = " ")
  
  return(unique_sentence)
}



# Function to find collocates and apply them to the text
combine_phrases <- function(text, collocation_list) {
  # Iterate over the phrases
  for (phrase in collocation_list) {
    # Replace the spaces in the phrase with a underscore
    combined_phrase <- gsub(" ", "_", phrase)
    
    # Replace the phrase in the text with the combined phrase
    # Add fixed = TRUE to treat the phrase as a literal string
    text <- gsub(phrase, combined_phrase, text, fixed = TRUE)
  }
  
  # Return the modified text
  return(text)
}

# TF-IDF function to extract keywords
get_keywords_tfidf <- function(df, text_column, id_column) {
  
  # Annotate the text using udpipe
  annotation <- udpipe_annotate(ud_model, x = df[[text_column]], doc_id = df[[id_column]])
  
  # Convert the annotation to a data frame
  annotation_df <- as.data.frame(annotation)
  
  # Filter the data frame to keep only the nouns and adjectives
  filtered_df <- annotation_df[annotation_df$upos %in% c("NOUN", "ADJ"),]
  
  # Convert the data frame to a tibble (a type of data frame that works well with tidytext)
  df2 <- as_tibble(filtered_df) %>%
    filter(!lemma %in% stop_words$word)  %>% # Remove stop words
    group_by(doc_id) %>%
    count(lemma, sort = TRUE) %>% # Count the frequency of each word for each document
    ungroup() %>%
    bind_tf_idf(lemma, doc_id, n) %>%
    filter(!(is.na(tf_idf))) # Compute TF-IDF scores
  
  # Combine all keywords for each Item
  df2 <- df2 %>%
    group_by(doc_id) %>%
    arrange(desc(tf_idf)) %>%
    summarize(keywords_tfidf = paste(lemma, collapse = " "))
  
  return(df2)
}

# Function to combine keywords
combine_keywords <- function(text, phrases_list) {
  # Split the text into words
  words <- strsplit(text, "/")[[1]]
  words <- trimws(words)  # trim whitespace
  
  # Iterate over the phrases in the list
  for (phrase in phrases_list) {
    # Check if the phrase exists in the text
    if (phrase %in% words) {
      # If it does, replace the spaces in the phrase with underscores
      words[which(words == phrase)] <- gsub(" ", "_", phrase)
    }
  }
  
  # Combine the words back into a single string
  new_text <- paste(words, collapse = "/")
  
  return(new_text)
}



# Function to calculate word count separated by "/"
word_count <- function(string) {
  words <- strsplit(string, split = "[/_[:space:]]")[[1]]
  length(words)
}

# Function to calculate Jaccard value
jaccard_value <- function(x, y) {
  x <- trimws(unlist(strsplit(x, " ")))
  y <- trimws(unlist(strsplit(y, " ")))
  length(intersect(x, y)) / length(union(x, y))
}

evaluate_model <- function(true_labels, predictions, predictions_prob) {
  
  true_labels <- factor(true_labels, levels=c(1, 0))
  predictions <- factor(predictions, levels=c(1, 0))
  
  # Classification report
  classification_report <- confusionMatrix(as.factor(true_labels), as.factor(predictions))
  print(classification_report)
  
  # Confusion matrix
  confusion_matrix <- table(true_labels, as.factor(predictions))
  
  # Accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  print(paste("Accuracy:", accuracy))
  
  # Precision and recall
  precision <- confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[2, 1])
  recall <- confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 2])
  print(paste("Precision:", precision))
  print(paste("Recall:", recall))
  
  # F1 score
  F1_score <- 2 * (precision * recall) / (precision + recall)
  print(paste("F1 Score:", F1_score))
  
  # ROC curve
  library(pROC)
  roc_obj <- roc(as.numeric(as.character(true_labels)), predictions_prob)
  #plot(roc_obj, main="ROC Curve", col="blue", lwd=2)
  
  # AUC
  auc_value <- auc(roc_obj)
  print(paste("AUC:", auc_value))
}


# # Define the text processing tokenizer
# tokenizer <- function(raw_text) {
#   # Remove punctuation
#   no_pun <- gsub("[[:punct:]]", "", raw_text)
#   no_pun
# }
# # Define a custom preprocessing function using the previously defined tokenizer
# preprocess <- content_transformer(function(x, tokenizer) unlist(strsplit(tokenizer(x), " ")))

# # optht function for LSA
# # The main optht function computes the optimal hard threshold for singular values in matrix denoising. 
# # It takes into account the aspect ratio of the matrix, singular values, and optionally, the noise level.
# optht <- function(beta, sv, sigma=NULL) {
#   # Compute aspect ratio of the input matrix
#   if (is.matrix(beta)) {
#     m <- min(dim(beta))
#     n <- max(dim(beta))
#     beta <- m / n
#   }
#   
#   # Check beta
#   if (beta < 0 || beta > 1) {
#     stop("Parameter `beta` must be in (0,1].")
#   }
#   
#   if (is.null(sigma)) {
#     # Sigma is unknown
#     cat("Sigma unknown.\n")
#     # Approximate w(beta)
#     coef_approx <- optimal_SVHT_coef_sigma_unknown(beta)
#     cat(paste0("Approximated `w(beta)` value: ", coef_approx, "\n"))
#     # Compute the optimal w(beta)
#     coef <- optimal_SVHT_coef_sigma_known(beta) / sqrt(median_marcenko_pastur(beta))
#   } else {
#     # Sigma is known
#     cat("Sigma known.\n")
#     # Compute optimal w(beta)
#     coef <- optimal_SVHT_coef_sigma_known(beta)
#   }
#   
#   # Compute cutoff (outside the conditional block)
#   cutoff <- coef * sqrt(length(sv)) * ifelse(is.null(sigma), 1, sigma)
#   
#   cat(paste0("`w(beta)` value: ", coef, "\n"))
#   cat(paste0("Cutoff value: ", cutoff, "\n"))
#   
#   # Compute and return rank
#   greater_than_cutoff <- which(sv > cutoff)
#   if (length(greater_than_cutoff) > 0) {
#     k <- max(greater_than_cutoff) + 1
#   } else {
#     k <- 0
#   }
#   
#   cat(paste0("Target rank: ", k, "\n"))
#   return(list(k=k, cutoff=cutoff))
# }
# 
# # This function implements Equation (11) in the referenced paper, 
# # calculating the optimal SVHT coefficient when the noise level σ is known.
# optimal_SVHT_coef_sigma_known <- function(beta) {
#   return(sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + sqrt(beta^2 + 14 * beta + 1))))
# }
# 
# # This function implements Equation (5) in the referenced paper, 
# # calculating the optimal SVHT coefficient when the noise level  σ is unknown.
# optimal_SVHT_coef_sigma_unknown <- function(beta) {
#   return(0.56 * beta^3 - 0.95 * beta^2 + 1.82 * beta + 1.43)
# }
# 
# # This function implements the Marcenko-Pastur distribution, 
# # used in the calculation of the median of the Marcenko-Pastur distribution.
# mar_pas <- function(x, topSpec, botSpec, beta) {
#   condition <- (topSpec - x) * (x - botSpec) > 0
#   result <- ifelse(condition, sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (2 * pi), 0)
#   return(result)
# }
# 
# # This function computes the median of the Marcenko-Pastur distribution, 
# # utilizing numerical integration and the previously defined _mar_pas function.
# median_marcenko_pastur <- function(beta) {
#   botSpec <- lobnd <- (1 - sqrt(beta))^2
#   topSpec <- hibnd <- (1 + sqrt(beta))^2
#   change <- 1
#   
#   while (change & ((hibnd - lobnd) > .001)) {
#     change <- 0
#     x <- seq(from=lobnd, to=hibnd, length.out=10)
#     y <- rep(0, length(x))
#     for (i in 1:length(x)) {
#       yi <- integrate(mar_pas, lower=x[i], upper=topSpec, topSpec=topSpec, botSpec=botSpec, beta=beta)$value
#       y[i] <- 1.0 - yi
#     }
#     
#     if (any(y < 0.5)) {
#       lobnd <- max(x[y < 0.5])
#       change <- 1
#     }
#     
#     if (any(y > 0.5)) {
#       hibnd <- min(x[y > 0.5])
#       change <- 1
#     }
#   }
#   
#   return((hibnd + lobnd) / 2)
# }

# Function to remove verbs from a piece of text
remove_verbs <- function(text, model) {
  # Annotate the text with POS tags
  annotation <- udpipe_annotate(model, x = text)
  tagged <- as.data.frame(annotation)
  
  # Filter out the verbs
  non_verbs <- tagged[tagged$upos != "VERB", ]
  
  # Reconstruct the text without verbs
  cleaned_text <- paste(non_verbs$lemma, collapse = " ")
  
  return(cleaned_text)
}