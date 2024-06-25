#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# In[9]:


nltk.download('punkt')
nltk.download('stopwords')


# In[10]:


def load_emails(data_directory):
    emails = []
    labels = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".txt"):
            with open(os.path.join(data_directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
                labels.append('spam' if 'spam' in filename else 'ham')
    return pd.DataFrame({'email': emails, 'label': labels})


# In[11]:


data_directory = r'C:\Users\himak\Downloads\training emails'
emails_df = load_emails(data_directory)
emails_df.head()


# In[12]:


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)


# In[13]:


emails_df['processed_email'] = emails_df['email'].apply(preprocess_text)


# In[14]:


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X = tfidf_vectorizer.fit_transform(emails_df['processed_email']).toarray()
y = emails_df['label']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# In[20]:


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# In[17]:


svm_predictions = svm_model.predict(X_test)
print("SVM Model")
print(classification_report(y_test, svm_predictions))
print("Accuracy:", accuracy_score(y_test, svm_predictions))


# In[18]:


nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Model")
print(classification_report(y_test, nb_predictions))
print("Accuracy:", accuracy_score(y_test, nb_predictions))


# In[21]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
svm_predictions = svm_model.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM Model')
plt.show()

