#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

from wordcloud import WordCloud

from bs4 import BeautifulSoup
from collections import Counter

import nltk

from nltk.corpus import stopwords

import re
import seaborn as sns
from PIL import Image
# create text with Markdown from within code cells¶
from IPython.display import Markdown as md 




# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[5]:


from gensim.models import word2vec


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn.metrics import roc_curve,roc_auc_score,auc

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[7]:


train_set = pd.read_csv('train.csv')


# In[8]:


pd.set_option('display.max_colwidth', -1)


# In[9]:


train_set.size


# In[10]:


train_set.head(15)


# ## Exploratory Data Analysis

# In[27]:


# Count target values by its factor
x = train_set.target.value_counts()
print("The amount disaster tweets is {}. And the amount for not disaster is {}.".
     format(x[1], x[0]))


# In[28]:


# Add title
plt.title("Amount of tweets - Disaster(1) or not(0)")
# Bar chart showing amount of both target values
sns.barplot(x.index,x)
# Add label for vertical axis
plt.ylabel("Count")
# Add label for hotizontal axis
plt.xlabel("Target")


# In[38]:


# Facet a plot by target column
g = sns.FacetGrid(train_set, col = 'target', height = 5, hue = 'target')
# Plot a histogram chart
g.map(sns.distplot, "num_words")
# Adjust title position
g.fig.subplots_adjust(top=0.8)
# Add general title
g.fig.suptitle('Distribution of number of words by target', fontsize=16)
# Set title to each chart
axes = g.axes.flatten()
axes[0].set_title("Not disaster")
axes[1].set_title("Disaster")


# In[39]:


plt.figure(figsize=(9,5))
# Add title
plt.title("Boxplot - Comparing distribution of number of words by target")
# Boxplot
sns.boxplot(x = "target", y = "num_words", hue="target", data = train_set)
# Add label for vertical axis
plt.ylabel("Number of Words")
# Add label for hotizontal axis
plt.xlabel("Target")


# In[12]:


test_set = pd.read_csv('test.csv')


# In[13]:


test_set.head()


# In[14]:


x_train = train_set.text

y_train = train_set['target']


# In[15]:


print(len(y_train))


# In[16]:


x_train.head()


# In[31]:


# Datasets shape
print('Train dataset:\n{} rows\n{} columns'.format(train_set.shape[0], train_set.shape[1]))
print('\nTest dataset:\n{} rows\n{} columns'.format(test_set.shape[0], test_set.shape[1]))


# In[32]:


proportion = x/train_set.shape[0] # Compute the tweets proportion by target
md("The percentual of disaster tweets is {}%, and {}% for not disaster.".
     format(round(proportion[1]*100,0),round(proportion[0]*100, 0)))


# In[33]:



fig1, ax1 = plt.subplots()
ax1.pie(proportion, 
        explode = (0, 0.1), # only "explode" the 2nd slice
        labels  = ['Not disaster', 'Disaster'], 
        autopct = '%1.1f%%',
        shadow = True, 
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Percentual of tweets")
plt.show()


# In[34]:


# Create a new feature with text lenght, or number of characters
train_set['length'] = train_set['text'].str.len()
# Create a new feature with number of words
train_set['num_words'] = train_set['text'].str.split().map(lambda x: len(x))
train_set.head(3)


# In[35]:


# Text length summary by target
train_set.groupby(['target']).length.describe()


# In[36]:


g = sns.FacetGrid(train_set, col = 'target', height = 5, hue = 'target')
# Plot a histogram chart
g.map(plt.hist, "length")
# Adjust title position
g.fig.subplots_adjust(top=0.8)
# Add general title
g.fig.suptitle('Text lenght by target', fontsize=16)
# Set title to each chart
axes = g.axes.flatten()
axes[0].set_title("Not disaster")
axes[1].set_title("Disaster")


# In[43]:


# Function to compute many unique words have this text
def counter_word (text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count


# In[45]:


text_values = train_set["text"]
counter = counter_word(text_values)
md("The training dataset has {} unique words".format(len(counter)))


# In[47]:


# Groups the top 20 keywords
x = train_set.keyword.value_counts()[:20]
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("20 hottest keyword in the text")
# Bar chart showing amount of both target values
sns.barplot(x.index, x, color="c")
# Add label for vertical axis
plt.ylabel("Count")
# Add label for hotizontal axis
plt.xlabel("Keywords")
# Rotate the label text for hotizontal axis
plt.xticks(rotation=90)


# In[48]:


# Groups the top 20 location
x = train_set.location.value_counts()[:20]
# Set the width and height of the figure
plt.figure(figsize=(10,6))
# Add title
plt.title("Top 20 location")
# Bar chart showing amount of both target values
sns.barplot(x.index, x, color = "pink")
# Add label for vertical axis
plt.ylabel("Count")
# Add label for hotizontal axis
plt.xlabel("Location")
# Rotate the label text for hotizontal axis
plt.xticks(rotation=90)


# ## Data Preprocessing 

# In[17]:


def clean_data(sentence, remove_stopwords = True,string = True):
    
    sentence = BeautifulSoup(sentence).get_text()
    
    sentence = re.sub("[^a-zA-Z]"," ",sentence)
    #sentence.replace(/^(?:https?:\/\/)?(?:www\.)?/i, "").split('/')[0]
    
    words = sentence.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        
        words  = [word for word in words if word not in stops and word not in ['http','https']]
        
    if string:
        
        return " ".join(words)
    
    else:
        
        return words
    
    


# In[70]:


train_set['text'] = train_set['text'].apply(lambda x: clean_data(x))
test_set['text'] = test_set['text'].apply(lambda x: clean_data(x))


# In[64]:


x_train = [clean_data(text) for text in x_train]


# In[58]:


#x_train = re.sub("[^a-zA-Z]"," ",train_set['text'][0])


# In[65]:


x_test = [clean_data(text) for text in test_set['text']]


# In[71]:


# Dictionary of abbreviations
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}


# In[76]:


def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

#def convert_abbrev_in_text(text):
 ### text = ' '.join(tokens)
   # return text


# In[78]:


# Appy abbreviation to text
x_train= train_set['text'].apply(lambda x: convert_abbrev(x))
x_test = test_set['text'].apply(lambda x: convert_abbrev(x))


# In[79]:


x_train[0:5]


# In[80]:


train_set["text"][0]


# In[81]:


text


# In[82]:


def show_wordcloud(text,title = None):
    
    wordcloud = WordCloud( background_color='black',
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1).generate(str(text))
    
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()
    


# In[83]:


show_wordcloud(x_train,title = " Most common words from the training datasets")


# In[84]:


show_wordcloud(x_test,title = " Most common words from the testing datasets")


# In[88]:


print(len(x_test))
print(type(x_test))


# In[91]:


print(len(x_train))
print(type(x_train))


# In[62]:


x1_train = x_train
print(len(x1_train))


# In[92]:


tfidfVectorizer = TfidfVectorizer(min_df=2,max_features= 5000)
tfidf_train_feat = tfidfVectorizer.fit_transform(x_train)
tfidf_test_feat =  tfidfVectorizer.transform(x_test)


# In[93]:


tfidfVectorizer.get_feature_names()


# In[96]:


first_vector_tfidfvectorizer=tfidf_train_feat[0]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidfVectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
 


# In[97]:


# Bag Of Word 
Vectorizer = CountVectorizer(analyzer="word",tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,
                            max_features = 5000)

train_feat = Vectorizer.fit_transform(x_train).toarray()

test_feat  = Vectorizer.transform(x_test).toarray()


# In[98]:


Vectorizer.vocabulary_


# In[99]:


x_train , x_val,y_train,y_val = train_test_split(train_feat,y_train,test_size =0.2)


# In[100]:


print(len(y_train))


# In[102]:


#tf_x_train , tf_x_val,y_train,y_val = train_test_split(tfidf_train_feat,y_train,test_size =0.2)


# In[103]:


print("length of x_train :{} and x_val :{} and y_train :{} and y_val :{}".format(len(x_train),len(x_val),len(y_train),len(y_val)))


# In[104]:



def predict(test_feat,train_feat,y_test,y_train,model,title = "Random forest"):
    
    model.fit(train_feat,y_train)
    
    preds = model.predict(test_feat)
    
    fpr,tpr,_ = roc_curve(y_test,preds)
    
    roc_auc = auc(fpr,tpr)
    
    print("AUC :",roc_auc)
    
    plt.plot(fpr,tpr)
    
    plt.title(title)
    
    plt.plot([0,1],[0,1],color = "navy",linestyle ="--")
    
    
    plt.xlabel("False positive Rate")
    
    plt.ylabel("True positive Rate")
    
    plt.show()
    
    return preds


# In[105]:


preds_rf = predict(x_val,x_train,y_val,y_train,RandomForestClassifier(n_estimators=100))
print(classification_report(preds_rf,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_rf,y_val))
preds_nb = predict(x_val,x_train,y_val,y_train,MultinomialNB(),title = "Naive Bayes")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[106]:


from sklearn.metrics import classification_report,confusion_matrix
# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[108]:


# Showing Confusion Matrix
plot_cm(y_val,preds_rf, 'Confution matrix of Tweets', figsize=(7,7))


# In[109]:


preds_nb = predict(x_val,x_train,y_val,y_train,LinearSVC(),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[110]:


# Showing Confusion Matrix
plot_cm(y_val,preds_nb, 'Confution matrix of Tweets', figsize=(7,7))


# In[111]:



preds_nb = predict(x_val,x_train,y_val,y_train,SVC(kernel="poly", degree=2, coef0=1, C=5),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[112]:


# Showing Confusion Matrix
plot_cm(y_val,preds_nb, 'Confution matrix of Tweets', figsize=(7,7))


# # Tf-idf 

# In[41]:


preds_nb = predict(tf_x_val,tf_x_train,y_val,y_train,MultinomialNB(),title = "Naive Bayes")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[42]:


preds_nb = predict(tf_x_val,tf_x_train,y_val,y_train,LinearSVC(),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[43]:


preds_nb = predict(tf_x_val,tf_x_train,y_val,y_train,SVC(kernel="poly", degree=2, coef0=1, C=5),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[69]:


embedding_dim = 200

tweets = [sentence.split() for sentence in x1_train]

print(len(tweets))
tweets[0:2]


# In[75]:


model1 = word2vec.Word2Vec(sentences =tweets,
                          size = embedding_dim,
                          window=5,
                          min_count=1,
                          sg = 1,
                          workers= 4,
                          negative= 10
                         
                         
                         )


# In[89]:


model.train(tweets, total_examples= len(x1_train), epochs=20)


# In[90]:


words = list(model1.wv.vocab)


# In[105]:


# save model 
filename = 'eathequake_tweets_word2vec.txt'

model1.wv.save_word2vec_format(filename,binary = False)


# In[91]:


print(len(words))


# In[85]:


model1.wv.most_similar(positive="allah")


# In[86]:


model1['allah']


# In[87]:


print(len(model1['earthquake']))


# In[95]:


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model1[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


# In[96]:


wordvec_arrays = np.zeros((len(tweets), 200))
for i in range(len(tweets)):
    wordvec_arrays[i,:] = word_vector(tweets[i], 200)
wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape


# In[104]:


wordvec_df


# In[113]:


word2vec_x_train , word2vec_x_val,y_train,y_val = train_test_split(wordvec_df,y_train,test_size =0.2)


# In[112]:


print(len(y_train))


# # word2_vec embedding 

# In[117]:


preds_rf = predict(word2vec_x_val,word2vec_x_train,y_val,y_train,RandomForestClassifier(n_estimators=100))
print(classification_report(preds_rf,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_rf,y_val))


# In[120]:


preds_nb = predict(word2vec_x_val,word2vec_x_train,y_val,y_train,LinearSVC(),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[121]:


preds_nb = predict(word2vec_x_val,word2vec_x_train,y_val,y_train,SVC(kernel="poly", degree=2, coef0=1, C=5),title = "SVM")
print(classification_report(preds_nb,y_val))
print("Accuracy:\n")
print(accuracy_score(preds_nb,y_val))


# In[ ]:




