
# coding: utf-8

# In[8]:


import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['labels', 'messages'])


# In[9]:


messages.head()


# In[10]:


messages.describe()


# In[11]:


messages.groupby('labels').describe()


# In[12]:


messages.head()
    


# In[13]:


messages['length'] = messages['messages'].apply(len)


# In[14]:


# messages.groupby('messages').describe()


# In[15]:


messages.head()


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


get_ipython().magic('matplotlib inline')

messages['length'].plot.hist(bins=50)


# In[18]:


messages['length'].describe()


# In[19]:


messages[messages['length'] == 910 ]['messages'].iloc[0]


# In[20]:


messages.hist(column='length', by='labels', bins=150, figsize=(12, 4))

# X axis shows sms size
# Y axis shows frequency of sms of that size


# In[21]:


import string


# In[22]:


list(string.punctuation)


# In[23]:


punc = '#This string is full -of crap and spam!! DaYM how do I handle this?'


# In[24]:


nopunc = [c for c in punc if c not in string.punctuation]
nopunc


# In[25]:


''.join(nopunc)


# In[26]:


from nltk.corpus import stopwords


# In[27]:


print(stopwords.words('english'))


# In[28]:


nopunc = ''.join(nopunc).split()
nopunc


# In[29]:


clean_text = [word for word in nopunc if word.lower() not in stopwords.words('english')]


# In[30]:


clean_text


# In[31]:


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
    
    


# In[32]:


messages.head()


# In[33]:


messages['messages'].head(5).apply(text_cleaner)


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer #It returns a vocabulary of words,
#then transform function is used to create a CountVector for every message 


# In[35]:


#Here we are providing the countvectorizer a clean list of words, cleaned using our text_cleaner function
bow_transformer = CountVectorizer(analyzer=text_cleaner).fit(messages['messages'])


# In[36]:


print(len(bow_transformer.vocabulary_))
bow_transformer.vocabulary_


# In[37]:


mess4 = messages['messages'][3]
print(mess4)


# 

# In[62]:


bow4 = bow_transformer.transform([mess4])
print(bow4)


# In[39]:


print(bow4.shape)


# In[ ]:





# In[40]:


bow_transformer.get_feature_names()[9554]


# In[41]:


messages_bow = bow_transformer.transform(messages['messages'])


# In[42]:


print("Shape of Sparse Matrix", messages_bow.shape)


# In[43]:


messages_bow.nnz #non-zero values


# In[44]:


#Fullness
sparsity = (100.0 * (messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1])))
print("Sparsity: ", round(sparsity))


# In[45]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[46]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[64]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[67]:


print(tfidf4)
tfidf4.shape


# In[69]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['unconscious']]


# In[89]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[ ]:


#Training the model


# In[51]:


from sklearn.naive_bayes import MultinomialNB


# In[83]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['labels'])


# In[88]:


print("Predicted label for 4th SMS: ",spam_detect_model.predict(tfidf4)[0])
print("Actual label for 4th SMS: ",messages['labels'][3])


# In[76]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[102]:


#Accuracy is 100% as we are testing on the same data on which we trained, so we need to split the dataset


# In[104]:


#Evaluating the model

from sklearn.metrics import classification_report
print(classification_report(messages['labels'], all_predictions))


# In[106]:


from sklearn.cross_validation import train_test_split


# In[93]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['messages'], messages['labels'], test_size=0.3)
print(len(msg_train), len(msg_test), len(msg_train)+ len(msg_test))


# In[94]:


#Creating a pipeline as the steps to be followed stays the same
from sklearn.pipeline import Pipeline


pipeline = Pipeline([
    ('bag_of_words', CountVectorizer(analyzer=text_cleaner)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[97]:


pipeline.fit(msg_train, label_train)


# In[98]:


predictions = pipeline.predict(msg_test)


# In[112]:


print(classification_report(predictions, label_test))

