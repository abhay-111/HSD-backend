#Importing the packages
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re





nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")






def clean (text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text



# def main():

data = pd.read_csv("hate_speech_dataset.csv")

#To preview the data
# print(data.head())
data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
# print(data.head())


data["tweet"] = data["tweet"].apply(clean)
print(data['labels'].head())
x = np.array(data["tweet"])
y = np.array(data["labels"])
cv = CountVectorizer()
X = cv.fit_transform(x)



#Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#Model building
model = DecisionTreeClassifier()


#Training the model
model.fit(X_train,y_train)


#Testing the model
y_pred = model. predict (X_test)
y_pred#Accuracy Score of our model
print (accuracy_score (y_test,y_pred))


# inp = '~'
# while inp != '\n':
    #Predicting the outcome
    
# while True:
#     inp = input("Enter the sample speech : ")
#     inp = cv.transform([inp]).toarray()
#     print(model.predict(inp))


def run(text) -> str:
    inp = text
    inp = cv.transform([inp]).toarray()
    result = model.predict(inp)
    print(result)
    return result




# if __name__ == '__main__':
#     main()
    

