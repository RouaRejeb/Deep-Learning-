SMS Spam Detection with NLP and XGBoost

Project Overview

This is a small machine-learning project I built to practice NLP from start to finish.

The goal was simple: take SMS messages, process the text, and train a model that can distinguish between ham (legitimate) and spam messages.

The project follows this workflow:

Raw SMS data → Cleaning → NLP preprocessing → TF-IDF → XGBoost → Predictions → Evaluation

I followed the NLP learning guide as a starting point, but I also took the time to look at the data and understand what the model's results actually mean rather than focusing only on the final score.

Dataset

The dataset contains SMS messages labelled as either ham or spam.

After removing duplicate messages, the dataset contains:

5,169 unique messages

4,516 ham messages

653 spam messages

The classes are not evenly distributed:

87.37% ham

12.63% spam

This matters because a high accuracy score on its own can be misleading when one class is much more common than the other. For this reason, I also looked at precision, recall, F1-score, and the confusion matrix.

NLP Preprocessing

The text was prepared through several steps:

Lowercasing the messages

Removing punctuation

Tokenizing the text with NLTK

Removing English stopwords

Applying stemming

Rejoining the processed tokens

The purpose was to turn the original SMS text into a cleaner representation that could be used for machine learning.

Feature Extraction

I explored text representation using Bag of Words and TF-IDF.

The final model uses TF-IDF, with the vocabulary limited to the 3,000 most important features.

TF-IDF converts the messages into numerical features while giving more importance to terms that help distinguish messages from one another.

Model

The classification model used in this project is XGBoost (XGBClassifier).

The model was trained on an 80/20 train-test split.

The trained model and prediction files were saved so that the final evaluation could be performed separately.

Results

The model achieved a test accuracy of 97.29%.

At first, this looks like a very strong result, but accuracy is not the only thing I wanted to look at because the dataset is imbalanced.

Classification metrics

Metric

Score

Accuracy

97.29%

Precision

93.98%

Recall

86.21%

F1-score

89.93%

What these results mean

Precision — 93.98%

When the model predicts that a message is spam, it is correct about 94% of the time.

For a spam filter, this is important because incorrectly marking a legitimate message as spam can be annoying or potentially cause an important message to be missed.

Recall — 86.21%

The model detected about 86% of the actual spam messages in the test set.

This also shows where the model has some room for error: it did not catch every spam message.

F1-score — 89.93%

The F1-score gives a balance between precision and recall. A score close to 90% suggests that the model is doing a good job overall at identifying spam while keeping false spam predictions relatively low.

Confusion Matrix

The test results give the following picture:

881 ham messages were correctly classified as ham.

125 spam messages were correctly classified as spam.

8 ham messages were incorrectly classified as spam.

20 spam messages were incorrectly classified as ham.

The two types of mistakes are worth looking at separately.

The 8 false positives mean that a small number of legitimate messages were treated as spam. The relatively high precision reflects that the model is fairly cautious about calling a message spam.

The 20 false negatives mean that some spam messages were missed. This is reflected in the lower recall compared with precision.

For me, this is more informative than simply saying that the model has 97.29% accuracy. It shows what the model is actually getting wrong.

Looking at the Predictions

The first predictions produced by the model included both ham and spam labels, for example:

[0 0 0 0 0 1 0 1 0 0]

where 0 represents ham and 1 represents spam.

Seeing the predictions this way was useful because it made the final step of the project more concrete: the model was no longer just producing a score, it was making a decision for each unseen SMS message.

My Takeaway

What I found most interesting about this project was seeing how much preparation is needed before a model can actually make useful predictions.

The XGBoost model itself is only one part of the process. Cleaning the messages, tokenizing them, removing stopwords, stemming the words, and converting the text into TF-IDF features all contribute to the final result.

The 97.29% accuracy is encouraging, but I would not use that number alone to judge the model. Looking at the confusion matrix helped me understand that the model correctly identified 125 spam messages, while missing 20, and incorrectly flagged 8 legitimate messages.

Overall, this project gave me a much better understanding of the complete NLP workflow and, more importantly, of how to interpret model predictions instead of just reporting a performance score.

What I Practiced

Python and Pandas

Data cleaning

Duplicate detection and removal

Exploratory data analysis

Class imbalance analysis

NLTK text preprocessing

Tokenization

Stopword removal

Stemming

Bag of Words

TF-IDF

Label encoding

Train/test splitting

XGBoost classification

Model evaluation

Confusion matrix analysis

Precision, recall, F1-score and accuracy

Saving and loading machine-learning models with NumPy and Joblib

Project Files

SMS-Spam-Detection/
│
├── spamprediction.ipynb
├── cleaned_spam_data.csv
├── X_features.npz
├── y_encoded.npy
├── y_test.npy
├── y_pred.npy
├── xgboost_spam_model.pkl
└── README.md

Note

This project was created as a learning and portfolio project while practicing NLP and machine learning. The results are specific to this dataset and the model configuration used in the notebook.
