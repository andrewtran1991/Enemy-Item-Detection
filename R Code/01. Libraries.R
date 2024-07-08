# set options
options(stringsAsFactors = F)
options(scipen = 999)
options(max.print=1000)

# required libraries
# General functions
library(ips.tools) # IPS Tools
library(tidyverse) # tidyverse: Collection of packages for data manipulation and visualization
library(readr) # readr: Used for reading rectangular text data
library(openxlsx) # openxlsx: Read, write and edit XLSX files
library(reshape2) # reshape2: Flexibly reshape data
library(doParallel) # For parallel processing
library(here) # To aid in the reading and writing of files
library(hashr) # provides hash functions to quickly summarize data 
library(stringr) # efficient manipulation and analysis of strings
library(stringi)
# NLP
library(tidytext) # tidytext: Text mining using tidy data principles
library(tm)  # tm: Framework for text mining applications
library(wordcloud) # wordcloud: Visualization tool for creating word clouds
library(udpipe) # udpipe: Tokenization, parts of speech tagging, lemmatization and dependency parsing
library(stringdist) # stringdist: Approximate string matching and string distance functions
library(textTinyR) # textTinyR: Text mining and word similarity
library(topicmodels) # topicmodels: Topic modeling of text documents
library(NLP) # For general NLP tasks
library(text2vec)# Load the required library for TF-IDF transformation
library(textstem) # is used for stemming and lemmatizing text
library(SnowballC) # This package provides functions for stemming and lemmatizing words.
library(stopwords) # stopwords
library(textmineR) # text mining in R, providing functions for document clustering, topic modeling, and more.
library(themis) # A collection of preprocessing strategies and sampling methods
#library(tidymodels) # A collection of packages for modeling and machine learning using tidyverse principles
library(lda) # This package provides a basic implementation of LDA.
library(ldatuning) # This package provides functions for tuning LDA models.
library(lsa) # LSA
library(irlba) # truncated SVD

# ML
library(caTools) # Provides functions for moving window statistics, binary classification
library(caret) # comprehensive library for training and evaluating classification 
library(smotefamily) # collection of oversampling techniques
library(pROC) # visualizing, smoothing, and comparing ROC curve
#library(vip) # Variable Importance Plots
#library(randomForest) # random forest for classification
library(e1071) # For SVM and other classifiers
library(caTools) # For splitting the dataset
library(Matrix) # For matrix operations
library(irlba) # For truncated SVD
library(caret) # For confusion matrix and other metrics
library(slam) # slam: Sparse Lightweight Arrays and Matrices for fast computation
library(randomForestSRC) # random forest for classification
library(RColorBrewer)

# library(fastLink) # fastLink: Implements fast probabilistic record linkage and deduplication
# library(FactoMineR) # FactoMineR: Multivariate Exploratory Data Analysis and Data Mining
# library(factoextra) # factoextra: Extract and visualize the results of multivariate data analyses
# library(flextable) # flextable: Functions for tabular reporting
# library(GGally) # GGally: Extension of 'ggplot2', provides functions to plot pairs
# library(ggdendro) # ggdendro: Tools for creating dendrograms with 'ggplot2'
# library(igraph) # igraph: Network analysis and visualization
# library(network) # network: Tools to create and modify network objects
# library(Matrix) # Matrix: Sparse and Dense Matrix classes and methods
# library(sna) # sna: Tools for Social Network Analysis
# library(kableExtra) # kableExtra: This package provides functions for formatting and exporting tables in R.
# library(DT) # This package provides functions for creating interactive tables in R.
# library(pals) # This package provides a palette of colors for word clouds.
# library(quanteda) # quanteda: Quantitative Analysis of Textual Data
# library(quanteda.textstats) # quanteda.textstats: Textual statistics in quanteda
# library(quanteda.textplots) # quanteda.textplots: Text plots in quanteda
# library(flextable) # This package provides functions for creating and formatting flextables.
