# Sentiment Classification using NLTK and Logistic Regression
![Python](https://img.shields.io/badge/Python-3.12.4-blueviolet)
![Tensorflow](https://img.shields.io/badge/ML-Tensorflow-fcba03)
![Colab](https://img.shields.io/badge/Editor-GColab-blue)


![sentiment_intro_img](sentiment.jpg)
This project demonstrates sentiment analysis on the Sentiment140 dataset, which contains 1.6 million tweets. The primary goal is to classify the sentiment of tweets as positive or negative using a logistic regression model.

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import zipfile
import pickle
```

## Setup

### Kaggle API Setup
1. Ensure Kaggle API credentials are set up:
    ```sh
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

### Download Dataset
1. Download the Sentiment140 dataset from Kaggle:
    ```sh
    !kaggle datasets download -d kazanova/sentiment140
    ```

### Extract Dataset
1. Unzip the dataset:
    ```python
    dataset_path = "/content/sentiment140.zip"
    with zipfile.ZipFile("sentiment140.zip", "r") as zip_ref:
        zip_ref.extractall()
        print("Done")
    ```

## Dataset Information

The Sentiment140 dataset contains 1,600,000 tweets annotated with sentiment labels:
- `0` = Negative
- `2` = Neutral
- `4` = Positive

### Dataset Fields
- `target`: Sentiment polarity
- `ids`: Tweet ID
- `date`: Date of the tweet
- `flag`: Query (if any)
- `user`: User who tweeted
- `text`: Content of the tweet

## Preprocessing

1. Import the dataset:
    ```python
    dataset = pd.read_csv('/content/train.csv', encoding="ISO-8859-1")
    ```

2. Rename columns and reload the dataset:
    ```python
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    dataset = pd.read_csv('/content/train.csv', names=col_names, encoding="ISO-8859-1")
    ```

3. Handle missing values:
    ```python
    dataset.isnull().sum()
    ```

4. Convert target labels (`4` to `1`):
    ```python
    dataset.replace({'target':{4:1}}, inplace=True)
    ```

## Text Preprocessing

1. Download NLTK stopwords:
    ```python
    nltk.download('stopwords')
    ```

2. Define a stemming function:
    ```python
    port_stem = PorterStemmer()

    def stemmer(content):
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    ```

3. Apply stemming:
    ```python
    dataset['stemmed_content'] = dataset['text'].apply(stemmer)
    ```

## Model Training

1. Split data into training and testing sets:
    ```python
    x = dataset['stemmed_content'].values
    y = dataset['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    ```

2. Convert text data to numerical data using TF-IDF Vectorizer:
    ```python
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    ```

3. Train the logistic regression model:
    ```python
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    ```

## Model Evaluation

1. Calculate accuracy on training data:
    ```python
    x_train_prediction = model.predict(x_train)
    training_data_accuracy = accuracy_score(y_train, x_train_prediction)
    print('Accuracy score of the training data:', training_data_accuracy)
    ```

2. Calculate accuracy on test data:
    ```python
    x_test_prediction = model.predict(x_test)
    test_data_accuracy = accuracy_score(y_test, x_test_prediction)
    print('Accuracy score of the test data:', test_data_accuracy)
    ```

## Save the Model

1. Save the trained model:
    ```python
    filename = 'twitter_final_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    ```

## Prediction

1. Test the model with new data:
    ```python
    x_new = x_test[200]
    prediction = model.predict(x_new)
    if prediction[0] == 0:
        print("Negative")
    else:
        print("Positive")
    ```
---
If you have any doubts regarding this project, do email me at the email given in my profile.
