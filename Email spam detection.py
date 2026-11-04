import pandas as pd

# Load the email dataset
data = pd.read_csv("emails.csv")

# Display first 5 records
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Count spam and ham emails
print("\nEmail Label Counts:")
print(data["label"].value_counts())

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

sample_email = "CONGRATULATIONS!!! You WON a FREE PRIZE!!!"

cleaned_email = clean_text(sample_email)

print("Original Email:", sample_email)
print("Cleaned Email:", cleaned_email)


def tokenize_text(text):
    words = text.split()
    return words

sample_email = "congratulations you won a free prize"

tokens = tokenize_text(sample_email)

print("Original Text:", sample_email)
print("Tokens:", tokens)

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(words):
    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return filtered_words

sample_words = ['you', 'won', 'a', 'free', 'prize']

result = remove_stopwords(sample_words)

print("Before:", sample_words)
print("After:", result)

from sklearn.feature_extraction.text import TfidfVectorizer

emails = [
    "win a free prize",
    "meeting scheduled tomorrow",
    "claim your free reward",
    "project meeting tomorrow"
]

vectorizer = TfidfVectorizer(stop_words="english")

email_vectors = vectorizer.fit_transform(emails)

print("Words:", vectorizer.get_feature_names_out())
print("TF-IDF Values:")
print(email_vectors.toarray())


from sklearn.model_selection import train_test_split

# Separate messages and labels
X = data["message"]
y = data["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))


from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Learn vocabulary from training data and convert it into numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

# Convert testing data using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)


from sklearn.naive_bayes import MultinomialNB

# Create Naive Bayes model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

print("Naive Bayes model trained successfully!")


from sklearn.svm import SVC

# Create SVM model
svm_model = SVC(kernel="linear")

# Train the model
svm_model.fit(X_train_tfidf, y_train)

print("SVM model trained successfully!")

# Predict using Naive Bayes
nb_predictions = nb_model.predict(X_test_tfidf)

# Predict using SVM
svm_predictions = svm_model.predict(X_test_tfidf)

print("Actual Labels:", list(y_test))

print("Naive Bayes Predictions:", nb_predictions)

print("SVM Predictions:", svm_predictions)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate Naive Bayes
print("\nNaive Bayes Evaluation")

print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Precision:", precision_score(y_test, nb_predictions, pos_label="spam", zero_division=0))
print("Recall:", recall_score(y_test, nb_predictions, pos_label="spam", zero_division=0))
print("F1 Score:", f1_score(y_test, nb_predictions, pos_label="spam", zero_division=0))


# Evaluate SVM
print("\nSVM Evaluation")

print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Precision:", precision_score(y_test, svm_predictions, pos_label="spam", zero_division=0))
print("Recall:", recall_score(y_test, svm_predictions, pos_label="spam", zero_division=0))
print("F1 Score:", f1_score(y_test, svm_predictions, pos_label="spam", zero_division=0))


from sklearn.metrics import confusion_matrix

# Naive Bayes Confusion Matrix
nb_cm = confusion_matrix(
    y_test,
    nb_predictions,
    labels=["ham", "spam"]
)

print("Naive Bayes Confusion Matrix:")
print(nb_cm)

# SVM Confusion Matrix
svm_cm = confusion_matrix(
    y_test,
    svm_predictions,
    labels=["ham", "spam"]
)

print("SVM Confusion Matrix:")
print(svm_cm)


# Take a new email from the user
new_email = input("Congratulations you won a free prize click here ")

# Convert email into TF-IDF features
new_email_tfidf = vectorizer.transform([new_email])

# Predict using Naive Bayes
nb_result = nb_model.predict(new_email_tfidf)

# Predict using SVM
svm_result = svm_model.predict(new_email_tfidf)

print("Naive Bayes Prediction:", nb_result[0])
print("SVM Prediction:", svm_result[0])