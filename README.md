# Enemy-Item-Detection
This project delves into the innovative integration of Machine Learning (ML) and Natural Language Processing (NLP) in refining enemy item detection, which could potentially enhance item selection, contribute to test validity, and support a more balanced assessment process.

### Introduction

Natural Language Processing (NLP) is an interdisciplinary domain of artificial intelligence focusing on the interaction between computers and humans through natural language. This project applies NLP in testing, particularly for enemy item detection in item banks. By leveraging various NLP techniques and classification algorithms, this study aims to optimize the detection of similar or redundant test items (enemy items) to assist subject matter experts (SMEs) in efficiently managing large item banks.

### Project Overview

This project is divided into two main stages:

1. **Analytical Stages**:
   - **Text Processing**: Involves cleaning text, tokenization, and extracting keywords and topics from a Document-Term Matrix.
   - **Similarity Index Calculation**: Utilizes methods such as Vector Space Model (VSM), Latent Semantic Analysis (LSA), and Latent Dirichlet Allocation (LDA) to compute similarity scores.
   - **Classification**: Classifies items as "enemies" based on similarity scores using Logistic Regression (LR) and Random Forest (RF) models.
   - **SME Review**: SMEs review and retrain the classification models as needed.

2. **Deployment Stages**:
   - **Report Generation**: Produces a report listing potential enemy item pairs.
   - **Application Development**: Creates an app for ECS/EPM/item writers to identify potential enemies for new pretest items.

### Methodology

**Data Preparation**:
- The VT bank was used, consisting of 1151 items (139 pretest ready and 1012 operational items).
- Text cleaning involved removing non-essential characters, punctuation, special characters, stopwords, and lemmatizing.
- The cleaned text was tokenized and converted to a document-term matrix using Bag of Words and Term Frequency-Inverse Document Frequency (TF-IDF) techniques.

**Similarity Calculation**:
- Each item was paired with every other item, resulting in 661,825 pairs.
- TF-IDF values were used to find keywords and calculate the Jaccard similarity index.
- Cosine similarity was calculated using both the VSM and LSA models.
- LDA was used to find latent topic distributions and calculate Jensen-Shannon divergence (JSD) values.
- Similarity indexes were combined with item attributes to train ML models.

**Model Training**:
- The dataset was split into 80% for training and 20% for testing.
- The classes were imbalanced, with only 582 pairs identified as enemies. SMOTE was applied to balance the classes.
- Both LR and RF models were trained for each NLP model (VSM, LSA, LDA).

### Results

- Enemy item pairs had higher cosine values for both VSM and LSA models compared to non-enemy pairs, while JSD values were lower for enemy item pairs.
- Models demonstrated high accuracy (>0.9), with LDA showing superior performance; however, accuracy's reliability is questioned due to class imbalance. SMOTE was applied to mitigate this issue.
- Trade-off observed between precision and recall, with many models showing high recall (0.33-0.87) but low precision (0.04-0.25), indicating a tendency to classify non-enemy pairs as enemy.
- The F1 Score, a balance of precision and recall, identified the LDA_RF model with a 0.8 threshold as having the most balanced performance among the tested models.

**Model Performance Metrics**:
- **Recall**: Proportion of actual positives correctly identified.
- **Precision**: Proportion of predicted positives that are actually positive.
- **Specificity**: Proportion of actual negatives correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.

**False Positives**:
- The LR model produced more false positives compared to the RF model.
- False positives were potential enemy item pairs for SME review.

### Discussion and Next Steps

- The project aimed to create a usable model to assist EPMs, ECSs, and item writers in identifying enemy item pairs.
- Current enemy detection procedures are manual; the project's outputs can streamline this process.
- Limitations include inconsistent enemy detection standards among ECSs and difficulties in processing image-based items.
- Future steps involve SME review of potential enemy pairs and model retraining based on new data.

### Contributions

Contributions are welcome. Please submit a pull request or open an issue for any suggestions or improvements.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

---