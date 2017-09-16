
# coding: utf-8

# In[3]:


import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['labels', 'messages'])


# In[4]:


messages.head()


# In[5]:


messages.describe()


# In[6]:


messages.groupby('labels').describe()


# In[7]:


messages.head()
    


# In[8]:


messages['length'] = messages['messages'].apply(len)


# In[9]:


# messages.groupby('messages').describe()


# In[10]:


messages.head()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


get_ipython().magic('matplotlib inline')

messages['length'].plot.hist(bins=50)


# In[13]:


messages['length'].describe()


# In[14]:


messages[messages['length'] == 910 ]['messages'].iloc[0]


# In[15]:


messages.hist(column='length', by='labels', bins=150, figsize=(12, 4))

# X axis shows sms size
# Y axis shows frequency of sms of that size


# In[16]:


import string


# In[17]:


list(string.punctuation)


# In[18]:


punc = '#This string is full -of stuff and spam!! DaYM how do I handle this?'


# In[19]:


nopunc = [c for c in punc if c not in string.punctuation]
nopunc


# In[20]:


''.join(nopunc)


# In[21]:


from nltk.corpus import stopwords


# In[22]:


print(stopwords.words('english'))


# In[23]:


nopunc = ''.join(nopunc).split()
nopunc


# In[24]:


clean_text = [word for word in nopunc if word.lower() not in stopwords.words('english')]


# In[25]:


clean_text


# In[26]:


def text_cleaner(mess):
    """
    Removing punctuations
    Removing stopwords
    returning list of clean words
    """
    nopunc_mess = ''.join([c for c in mess if c not in string.punctuation])
    
    nopunc_mess = nopunc_mess.split()
    
    cleantext_mess = [word for word in nopunc_mess if word.lower() not in stopwords.words('english')]
    
    return cleantext_mess
    
    


# In[27]:


messages.head()


# In[28]:


messages['messages'].head(5).apply(text_cleaner)


# In[29]:


from sklearn.feature_extraction.text import CountVectorizer #It returns a vocabulary of words,
#then transform function is used to create a CountVector for every message 


# In[30]:


#Here we are providing the countvectorizer a clean list of words, cleaned using our text_cleaner function
bow_transformer = CountVectorizer(analyzer=text_cleaner).fit(messages['messages'])


# In[31]:


print(len(bow_transformer.vocabulary_))


# In[32]:


mess4 = messages['messages'][3]
print(mess4)


# 

# In[33]:


bow4 = bow_transformer.transform([mess4])
print(bow4)


# In[34]:


print(bow4.shape)


# In[ ]:





# In[ ]:


bow_transformer.get_feature_names()[9554]


# In[ ]:


messages_bow = bow_transformer.transform(messages['messages'])


# In[ ]:


print("Shape of Sparse Matrix", messages_bow.shape)


# In[ ]:


messages_bow.nnz #non-zero values


# In[ ]:


#Fullness
sparsity = (100.0 * (messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1])))
print("Sparsity: ", round(sparsity))


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[ ]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[ ]:


print(tfidf4)
tfidf4.shape


# In[ ]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['unconscious']]


# In[ ]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[ ]:


#Training the model


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['labels'])


# In[ ]:


print("Predicted label for 4th SMS: ",spam_detect_model.predict(tfidf4)[0])
print("Actual label for 4th SMS: ",messages['labels'][3])


# In[ ]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[ ]:


#Accuracy is 100% as we are testing on the same data on which we trained, so we need to split the dataset


# In[ ]:


#Evaluating the model

from sklearn.metrics import classification_report
print(classification_report(messages['labels'], all_predictions))


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['messages'], messages['labels'], test_size=0.3)
print(len(msg_train), len(msg_test), len(msg_train)+ len(msg_test))


# In[ ]:


#Creating a pipeline as the steps to be followed stays the same
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bag_of_words', CountVectorizer(analyzer=text_cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[ ]:


pipeline.fit(msg_train, label_train)


# In[ ]:


predictions = pipeline.predict(msg_test)


# In[ ]:


print(classification_report(predictions, label_test))

