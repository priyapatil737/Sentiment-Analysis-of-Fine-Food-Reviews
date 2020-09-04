import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

st.title("Sentiment Analysis of Fine Food reviews")
st.sidebar.title("Sentiment Analysis of Fine Food reviews")

st.markdown("This application is a Streamlit dashboard to analyze the sentiment of reviews")



@st.cache(persist=True)

def load_data():
    df = pd.read_csv("/content/Reviews.csv")
   
    return df

df = load_data()
df=df.drop(['Time','Summary','Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator'],axis=1)
import seaborn as sns
import matplotlib.pyplot as plt

# we will create two classes i.e (1) for score >3 and (0) for score <3
# we will remove rows with score=3 as we are considering them to be neutal reviews
df = df[df['Score'] != 3] 
df['Score']=np.where(df['Score']>3,1,0)

#tokenization
import nltk
nltk.download('punkt')
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
def text_cleaning(text):
  # to remove special chars
  text= text.lower()
  text= re.sub('\[.*?\]', '', text)
  text = re.sub("\\W"," ",text) 
  # to remove links
  text = re.sub('https?://\S+|www\.\S+', '', text)
  #to remove punctuations
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text
df['Text']=df['Text'].apply(text_cleaning)
# TOKENIZATION
# We can tokenize a sentence by making use of a tokenizer.
df['tokenized_sentences'] = df['Text'].apply(nltk.word_tokenize)
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(nltk.corpus.stopwords.words())

def stopwords_cleaned(sentence):
    res = []
    for word in sentence:
        if word not in stopwords:
            res.append(word)
    return res
df['tokenized_removed_stopwords'] = df['tokenized_sentences'].apply(stopwords_cleaned)
from wordcloud import WordCloud, STOPWORDS
text_words = '' 
stopwords=set(STOPWORDS)
stopwords.update(["br", "href", "amazon", "food","gp","ve","grocery","store","although","suscribe","though","think","thought","maybe"])

#wordcloud
text = " ".join(review for review in df.Text)
  
wordcloud = WordCloud(width = 800, height = 800, background_color ='black', stopwords = stopwords, 
                min_font_size = 10).generate(text) 
                      
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud,interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 


#positive and negative word cloud
# Calculate the word frequencies and word energies
word_frequency = {} 
word_energy = {}
# Extracting the data
cleaned_texts = df['tokenized_removed_stopwords'].values
labels = df['Score'].values

for text, label in zip(cleaned_texts, labels):
    for word in text:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
            
        if label == 1:
            if word not in word_energy:
                word_energy[word] = 1
            else:
                word_energy[word] += 1
        else:
            if word not in word_energy:
                word_energy[word] = -1
            else:
                word_energy[word] -= 1
# Normalization
for word in word_energy:
    word_energy[word] /= word_frequency[word]

# removing unreliable words
reliable_word_energy = {}
for word in word_energy:
    # we assume that the energies would be reliable only for words that appear more than 500 times in our corpus. 
    if word_frequency[word] > 500: 
        reliable_word_energy[word] = word_energy[word]


top_50_positive_words = [v[0] for v in sorted(reliable_word_energy.items(), key=lambda x: x[1], reverse=True)[:50]]
top_50_negative_words = [v[0] for v in sorted(reliable_word_energy.items(), key=lambda x: x[1], reverse=False)[:50]]

#POSITIVE


from wordcloud import WordCloud
from matplotlib import pyplot as plt
#%matplotlib inline

def plot_word_clouds(keywords):
    wordcloud = WordCloud().generate(' '.join(keywords))
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

# word cloud visualization of top 50 positive words
st.title("WORDCLOUD for Positive Reviews")
plot_word_clouds(top_50_positive_words)


#NEGATIVE
# word cloud visualization of top 50 negative words
st.title("WORDCLOUD for Negative Reviews")
plot_word_clouds(top_50_negative_words)

def grade(x):
  if x==1:
    return "Positive"
  elif x==0:
    return "Negative"
    
df['Sentiment'] = df['Score'].apply(grade)
sns.countplot(df['Sentiment'])
st.pyplot()
y=df['Sentiment'].values
x=df['Text'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.2)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x_train,y_train)
y_pred = text_model.predict(x_test)
#Accuracy
from sklearn.metrics import accuracy_score
st.write("Accuracy of the model:")
accuracy_score(y_pred,y_test)*100
#Confusion Matrix
from sklearn.metrics import confusion_matrix
st.write("Confusion Matrix:")
st.table(confusion_matrix(y_pred,y_test))
#Classification Report
from sklearn.metrics import classification_report
st.write("Classification Report:")
st.markdown(classification_report(y_pred,y_test))