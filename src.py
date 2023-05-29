


import re
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
# You can replace this with your own dataset
df = pd.read_csv('hate_speech_dataset.csv', encoding='utf-8')

# Clean and preprocess the text data
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords])  # Remove stopwords
    return text


#To preview the data
# print(df.head())
df["labels"] = df["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
df = df[["tweet", "labels"]]
# print(df.head())


df['tweet'] = df['tweet'].apply(preprocess_text)

# Split the dataset into train and test sets
X = df['tweet']
y = df['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Bag-of-Words model
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_bow, y_train)

# Predict the test set labels
y_pred = clf.predict(X_test_bow)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# Example usage
# input_sentence = "women are wifes and wifes are hoes"
# while True :
#     input_sentence = input("Enter the text : ")
#     input_sentence_bow = vectorizer.transform([preprocess_text(input_sentence)])
#     hate_speech_prob = clf.predict_proba(input_sentence_bow)[:, 1].item()
#     print(f"Hate speech probability: {1 - hate_speech_prob:.4f}")



def run(input_text):
    # input_sentence = input("Enter the text : ")
    input_sentence = input_text
    input_sentence_bow = vectorizer.transform([preprocess_text(input_sentence)])
    hate_speech_prob = clf.predict_proba(input_sentence_bow)[:, 1].item()
    result = f"Hate speech probability: {1 - hate_speech_prob:.4f}"
    return result
