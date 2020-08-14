#!/usr/bin/env python
# coding: utf-8

# In[150]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[151]:


train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
train_original = train.copy()


# In[152]:


train_original


# In[153]:


test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
test_original = test.copy()


# In[154]:


test   #no label present for test data


# In[155]:


combine = train.append(test, ignore_index=True, sort=True)


# In[156]:


combine.head()


# In[157]:


combine.tail()


# In[158]:


def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    
    for i in r:
        text = re.sub(i, "", text)
        
    return text    


# In[159]:


combine['tidy_tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
# @[\w]* picks up any word starting with @ 
combine.head()


# In[160]:


combine['tidy_tweets'] = combine['tidy_tweets'].str.replace("[^a-zA-Z#]", " ")
# replace everything except character and hashtags with spaces
combine.head(10)


# In[161]:


combine['tidy_tweets'] = combine['tidy_tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combine.head()


# In[162]:


tokenized_tweet = combine['tidy_tweets'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[163]:


from nltk import PorterStemmer
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()


# In[164]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combine['tidy_tweets'] = tokenized_tweet
combine.head()


# In[165]:


from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import urllib
import requests


# In[166]:


all_words_positive = ' '.join(text for text in combine['tidy_tweets'][combine['label']==0])


# In[167]:


Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_positive)


# In[168]:


plt.figure(figsize=(10,20))
plt.imshow(wc.recolor(color_func=image_colors), interpolation="hamming")

plt.axis('off')
plt.show()


# In[169]:


all_words_negative = ' '.join(text for text in combine['tidy_tweets'][combine['label']==1])


# In[170]:


Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
image_colors = ImageColorGenerator(Mask)
wc = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_negative)


# In[171]:


plt.figure(figsize=(10,20))
plt.imshow(wc.recolor(color_func=image_colors), interpolation="gaussian")

plt.axis('off')
plt.show()


# In[172]:


def Hashtags_Extract(x):
    hashtags=[]
    
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
        
    return hashtags       


# In[173]:


ht_positive = Hashtags_Extract(combine['tidy_tweets'][combine['label']==0])

ht_positive


# In[174]:


ht_positive_unnest = sum(ht_positive, [])


# In[175]:


ht_positive_unnest


# In[176]:


ht_negative = Hashtags_Extract(combine['tidy_tweets'][combine['label']==1])
ht_negative


# In[177]:


ht_negative_unnest = sum(ht_negative, [])


# In[178]:


ht_negative_unnest


# In[179]:


# counting the frequency of the word having positive sentiment
word_freq_positive = nltk.FreqDist(ht_positive_unnest)
word_freq_positive


# In[180]:


# Creating dataframe for the most frequently used words in hashtags
df_positive = pd.DataFrame({
    'Hashtags': list(word_freq_positive.keys()),
    'Count': list(word_freq_positive.values())
})

df_positive.head(10)


# In[181]:


df_positive_plot = df_positive.nlargest(20, columns='Count')
sns.barplot(data=df_positive_plot, y='Hashtags', x='Count')
sns.despine()


# In[182]:


word_freq_negative = nltk.FreqDist(ht_negative_unnest)
word_freq_negative


# In[183]:


df_negative = pd.DataFrame({
    'Hashtags': list(word_freq_negative.keys()),
    'Count': list(word_freq_negative.values())
})

df_negative.head(10)


# In[184]:


df_negative_plot = df_negative.nlargest(20, columns='Count')
sns.barplot(data=df_negative_plot, y='Hashtags', x='Count')
sns.despine()


# In[185]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=10000, stop_words='english')

bow = bow_vectorizer.fit_transform(combine['tidy_tweets'])   #bag of words model
df_bow = pd.DataFrame(bow.todense())
df_bow


# In[186]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.90, min_df = 2, max_features = 10000, stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(combine['tidy_tweets'])   # TD-IDF model

df_tfidf = pd.DataFrame(tfidf_matrix.todense())
df_tfidf


# In[187]:


train_bow = bow[:31962]
train_bow.todense()


# In[188]:


train_tfidf_matrix = tfidf_matrix[:31962]
train_tfidf_matrix.todense()


# In[189]:


from sklearn.model_selection import train_test_split
# bag of words features
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow, train['label'], test_size = 0.3, 
                                                                      random_state = 2)


# In[190]:


#TF-IDF Features
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix, train['label'], test_size = 0.3, 
                                                                      random_state = 17)


# In[191]:


from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
Log_Reg = LogisticRegression(random_state=0, solver='lbfgs')


# In[192]:


Log_Reg.fit(x_train_bow, y_train_bow)


# In[193]:


prediction_bow = Log_Reg.predict_proba(x_valid_bow)
#Predicting the probabilities for a tweet falling into either Positive or Negative class.
prediction_bow


# In[194]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3

#converting the results to integer type
prediction_int = prediction_int.astype(np.int)
prediction_int


# In[195]:


# calculating f1 score

log_bow = f1_score(y_valid_bow, prediction_int)
log_bow


# In[196]:


# TF-IDF Features
Log_Reg.fit(x_train_tfidf, y_train_tfidf)


# In[197]:


# Predicting the probabilities
prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
prediction_tfidf


# In[198]:


prediction_int = prediction_tfidf[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
prediction_int


# In[199]:


# Calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)
log_tfidf


# In[200]:


from xgboost import XGBClassifier


# In[201]:


model_bow = XGBClassifier(random_state=22, learning_rate=0.9)
model_bow.fit(x_train_bow, y_train_bow)


# In[202]:


xgb = model_bow.predict_proba(x_valid_bow)
#Predicting the probability of a tweet falling into either Positive or Negative class.
xgb


# In[203]:


xgb = xgb[:,1]>=0.3
# converting the results to integer type
xgb_int = xgb.astype(np.int)
# calculating f1 score


# In[204]:


xgb_bow = f1_score(y_valid_bow, xgb_int)
xgb_bow


# In[205]:


model_tfidf = XGBClassifier(random_state=29, learning_rate=0.7)


# In[206]:


model_tfidf.fit(x_train_tfidf, y_train_tfidf)


# In[207]:


xgb_tfidf = model_tfidf.predict_proba(x_valid_tfidf)
xgb_tfidf


# In[208]:


xgb_tfidf = xgb_tfidf[:,1]>=0.3
xgb_int_tfidf = xgb_tfidf.astype(np.int)
xgb_int_tfidf


# In[209]:


score = f1_score(y_valid_tfidf, xgb_int_tfidf)
score


# In[210]:


from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)


# In[211]:


dct.fit(x_train_bow, y_train_bow)
dct_bow = dct.predict_proba(x_valid_bow)
dct_bow


# In[212]:


dct_bow = dct_bow[:,1]>=0.3
dct_int_bow = dct_bow.astype(np.int)
#calculating f1 score
dct_score_bow = f1_score(y_valid_bow, dct_int_bow)
dct_score_bow


# In[213]:


# tf-idf
dct.fit(x_train_tfidf, y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)
dct_tfidf


# In[214]:


dct_tfidf = dct_tfidf[:,1]>=0.3
dct_int_tfidf = dct_tfidf.astype(np.int)
dct_score_tfidf = f1_score(y_valid_tfidf, dct_int_tfidf)
dct_score_tfidf


# In[215]:


Algo_1 = ['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)']
score_1 = [log_bow, xgb_bow, dct_score_bow]
compare_1 = pd.DataFrame({
    'Model': Algo_1,
    'F1_Score': score_1
},
index = [i for i in range(1, 4)])
compare_1.T


# In[216]:


plt.figure(figsize=(18,5))
sns.pointplot(x='Model', y='F1_Score', data=compare_1)
plt.title('Bag-of-Words')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[217]:


Algo_2 = ['LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']
score_2 = [log_tfidf,score,dct_score_tfidf]
compare_2 = pd.DataFrame({
    'Model': Algo_2,
    'F1_Score': score_2
}, index=[i for i in range(1,4)])
compare_2.T


# In[218]:


plt.figure(figsize=(18,5))
sns.pointplot(x='Model', y='F1_Score', data=compare_2)
plt.title('TF-IDF')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[219]:


Algo_best = ['LogisticRegression(Bag-of-Words)','LogisticRegression(TF-IDF)']
score_best = [log_bow, log_tfidf]
compare_best = pd.DataFrame({
    'Model':Algo_best,
    'F1_Score':score_best
}, index=[i for i in range(1,3)])
compare_best.T


# In[220]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare_best)

plt.title('Logistic Regression(Bag-of-Words & TF-IDF)')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# In[222]:


test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1]>=0.3
test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label']]
submission.to_csv('result.csv', index=False)


# In[223]:


res = pd.read_csv('result.csv')
res


# In[ ]:




