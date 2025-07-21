# Spam Ham Classification Using TF-IDF and Machine Learning

This project demonstrates how to classify SMS messages as "spam" or "ham" (not spam) using Natural Language Processing (NLP) techniques and machine learning algorithms in Python.

## Features

- Data cleaning and preprocessing (tokenization, stemming, stopword removal)
- Feature extraction using Bag of Words (BoW) and TF-IDF
- Classification using Multinomial Naive Bayes
- Model evaluation with accuracy and classification report

## Dataset

The dataset used is the [SMSSpamCollection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which contains labeled SMS messages.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

Install dependencies using:

```bash
pip install pandas numpy scikit-learn nltk
```

## Usage

1. **Clone the repository and navigate to the project directory.**
2. **Download the SMSSpamCollection dataset** and place it in a folder named `smsspamcollection/` in the project root.
3. **Run the Jupyter Notebook:**

```bash
jupyter notebook "Spam Ham Classification Using TF-IDF And ML.ipynb"
```

4. **Follow the notebook cells to:**
   - Load and preprocess the data
   - Extract features using Bag of Words
   - Train and evaluate the classifier

## Code Overview

### Data Loading

```python
import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
```

### Data Preprocessing

- Remove non-alphabetic characters
- Convert to lowercase
- Remove stopwords
- Apply stemming

### Feature Extraction

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, ngram_range=(1,2))
X = cv.fit_transform(corpus).toarray()
```

### Model Training

```python
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

The notebook displays the accuracy and a detailed classification report for the spam/ham classifier.
