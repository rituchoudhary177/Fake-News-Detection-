#!/usr/bin/env python
# coding: utf-8

# In[27]:


## Fake News Detection and Analysis using Machine Learning algorithms and Hybrid classifiersFake News Detection and Analysis using Machine Learning algorithms and Hybrid classifiers ##
## Importing all the essential libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim
import gensim
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict, Counter
import string
from sklearn.feature_extraction.text import CountVectorizer


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Uploading The Dataset 
dataset = pd.read_csv('C:/Users/Ritu/Desktop/fake news detection/Kaggle dataset/train.csv')
dataset.head()


# In[3]:


dataset['label'].value_counts()


# In[4]:


# Dataset description
dataset.describe()


# In[5]:


dataset.index


# In[6]:


# Analysing and visualizing the dataset is balanced or not
dataset[['label']].hist(bins = 3)
plt.bar(np.arange(len([0,1])), dataset.groupby(['label']).size().values, 0.9,  color="orange")
plt.xticks(np.arange(len([0,1])), ['Real','Fake'])
plt.show()


# In[7]:


#  The Number Of Characters Present in Each Title by label
dataset['title'].str.len().hist(by=dataset['label'])


# In[8]:


#  The Number of Characters Present in Each Message by Type 
dataset['text'].str.len().hist(by=dataset['label'])


# In[9]:


# Function for Word Cloud
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()


# In[10]:


# Remove stopwords from title and text by label
articles_fake=dataset[dataset.label==1].copy()
articles_real=dataset[dataset.label==0].copy()
 


# In[11]:


# Word cloud for corpus_title_fake
show_wordcloud(articles_fake['title'])


# In[12]:


# Word cloud for corpus_title_not_fake
show_wordcloud(articles_real['title'])


# In[13]:


# Word cloud for corpus_text_fake
show_wordcloud(articles_fake['text'])


# In[14]:


# Word cloud for corpus_text_not_fake
show_wordcloud(articles_real['text'])


# In[1]:


#################################### Importing all the essential Libraries  ##############################################
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where


# In[2]:


##################################### Uploading DATASET  #######################################################################
dataset =pd.read_csv('C:/Users/Ritu/Desktop/fake news detection/Kaggle dataset/train.csv')


# In[3]:


dataset.head()


# In[4]:


################Independent Features of Dataset########################################################### 
x = dataset.drop('label',axis = 1)


# In[5]:


dataset['label']


# In[6]:


#####################################Displaying the Independent Features###################################################
x.head()


# In[7]:


######################################Dependent Feature(Target Feature)#####################################################
y = dataset['label']


# In[8]:


dataset.shape


# In[9]:


############################Data Preprocessing using LDA and TFIDF##################################################
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[10]:


######################Removing the Null Values from the Dataset#################################################### 
dataset = dataset.dropna()


# In[11]:


dataset.head()


# In[12]:


################################Dataset_After_Removing_Nan_Values(Missing Values from the dataset)#################### 
dataset.shape


# In[13]:


articles = dataset.copy()


# In[14]:


articles.reset_index(inplace = True)


# In[15]:


articles['text'][6]


# In[16]:


######################################Performing LDA and Data Cleaning(Latent Dirichlet allocation)############################################################
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer # Performing Stemming
ps = PorterStemmer()
list_articles = []
for i in range(0, len(articles)):
    print(i)
    review = re.sub('[^a-zA-Z]',' ', articles['text'][i]) #removing regular expression
    review = review.lower()  #Converting data into Lowercase 
    review = review.split()  
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    list_articles.append(review)   


# In[17]:


#Displaying the List developed after cleaning the data  
list_articles


# In[18]:


##Converting Text to Feature Vectors(Tokens)
## using tfidf Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer(max_features = 5000, ngram_range = (1,3))
features = tfidf_v.fit_transform(list_articles).toarray()


# In[19]:


x.shape


# In[20]:


y = articles['label']


# In[21]:


###Dividing the features into trainset and testset  
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(features, y, test_size=0.25,random_state=0)


# In[22]:


tfidf_v.get_feature_names()


# In[23]:


##
count_df = pd.DataFrame(x_train, columns=tfidf_v.get_feature_names())


# In[24]:


count_df.head()


# In[25]:


########################################Plotting Confusion Matrix#############################################################
def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    
    if normalize:
        cm = cm.astype('float')/ cm.sum(axis=1)[:,np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print ("confusion matrix")
        
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i,j]> thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("predicted label")                              
          


# In[44]:


###################################Applying ensemble Learning Algorithms######################################################


# In[45]:


## CREATING THE NAIVE BAYES CLASSIFICATION


# In[26]:


from sklearn import metrics
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score


# In[35]:


from sklearn import metrics
import numpy as np
import itertools

from sklearn.naive_bayes import MultinomialNB
NaiveBayes_Model = MultinomialNB().fit(x_train, y_train)


# In[ ]:


############### Measuring the performance of the Naive Bayes Model#######################


# In[36]:


predictedNB = NaiveBayes_Model.predict(x_test)

score = metrics.accuracy_score(y_test, predictedNB)
print(f'Accuracy: {round(score*100,2)}%')

precision = precision_score(y_test,predictedNB)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predictedNB)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predictedNB)
print(f'f1_score: {round(f1*100,2)}%')


cm = metrics.confusion_matrix(y_test, predictedNB)
plot_confusion_matrix(cm, classes=['FAKE','REAL'])


# In[ ]:


################################# HYBRID CLASSIFIER ##############################################


# In[53]:


##################################Importing libraries for AdaBoost Technique##################################
#ADABoost classifier
#importing essential Libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings 


# In[54]:


########################### CREATING THE AdaBoost-NAIVE BAYES CLASSIFICATION ################################


# In[58]:


########################### Model01 - Naive Bayes (Experiment-1)
NaiveBayes_Model1 = MultinomialNB()
AdaModel = AdaBoostClassifier(n_estimators=220,base_estimator= NaiveBayes_Model1, learning_rate = 1).fit(x_train, y_train)


# In[56]:


############### Measuring the performance of the AdaBoost-NaiveBayes Model#######################


# In[59]:


#Predict the response 
predicted_AdaBoostNB = AdaModel.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostNB))

score_ANB = metrics.accuracy_score(y_test, predicted_AdaBoostNB)
print(f'Accuracy: {round(score_ANB*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostNB)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostNB)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostNB)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostNB)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[88]:


##################### Model02 - Naive Bayes (Experiment-2)
NaiveBayes_Model2 = MultinomialNB()
AdaModelNB2 = AdaBoostClassifier(n_estimators=100,base_estimator= NaiveBayes_Model2, learning_rate = 1).fit(x_train, y_train)

#Predict the response 
predicted_AdaBoostNB2 = AdaModelNB2.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostNB2))

score_ANB2 = metrics.accuracy_score(y_test, predicted_AdaBoostNB2)
print(f'Accuracy: {round(score_ANB2*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostNB2)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostNB2)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostNB2)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostNB2)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[89]:


##################### Model03 - Naive Bayes (Experiment-3)

NaiveBayes_Model3 = MultinomialNB()
AdaModelNB3 = AdaBoostClassifier(n_estimators=150,base_estimator= NaiveBayes_Model3, learning_rate = 1).fit(x_train, y_train)

#Predict the response 
predicted_AdaBoostNB3 = AdaModelNB3.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostNB3))

score_ANB3 = metrics.accuracy_score(y_test, predicted_AdaBoostNB3)
print(f'Accuracy: {round(score_ANB3*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostNB3)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostNB3)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostNB3)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostNB3)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[90]:


##################### Model04 - Naive Bayes (Experiment-4)

NaiveBayes_Model4 = MultinomialNB()
AdaModelNB4 = AdaBoostClassifier(n_estimators=200,base_estimator= NaiveBayes_Model4, learning_rate = 1).fit(x_train, y_train)

#Predict the response 
predicted_AdaBoostNB4 = AdaModelNB4.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostNB4))

score_ANB4 = metrics.accuracy_score(y_test, predicted_AdaBoostNB4)
print(f'Accuracy: {round(score_ANB4*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostNB4)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostNB4)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostNB4)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostNB4)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[91]:


##################### Model05 - Naive Bayes (Experiment-5)

NaiveBayes_Model5 = MultinomialNB()
AdaModelNB5 = AdaBoostClassifier(n_estimators=250,base_estimator= NaiveBayes_Model5, learning_rate = 1).fit(x_train, y_train)

#Predict the response 
predicted_AdaBoostNB5 = AdaModelNB5.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostNB5))

score_ANB5 = metrics.accuracy_score(y_test, predicted_AdaBoostNB5)
print(f'Accuracy: {round(score_ANB5*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostNB5)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostNB5)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostNB5)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostNB5)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[98]:


## Comparing the results of various experiments on Naive bayes Model
ACCURACYNB = np.vstack((score,score_ANB2,score_ANB3,score_ANB4,score_ANB,score_ANB5))
model_number = np.array([1,2,3,4,5,6])
print('Model number Represents:')
print(' 1 = NaiveBayes classifier')
print(' 2 = AdaBoost_NaiveBayes with n_estimator-100')
print(' 3 = AdaBoost_NaiveBayes with n_estimator-150')
print(' 4 = AdaBoost_NaiveBayes with n_estimator-200')
print(' 5 = AdaBoost_NaiveBayes with n_estimator-220')
print(' 6 = AdaBoost_NaiveBayes with n_estimator-250')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model_number,ACCURACYNB, color = 'b', marker = '*', linewidth = 5)
for j in range(0,len(ACCURACYNB)):
              ax.annotate('%3f'%(ACCURACYNB[j]), (model_number[j],ACCURACYNB[j]))
plt.xlabel('MODELS')
plt.ylabel('ACCURACY')


# In[ ]:





# In[ ]:





# In[ ]:


###################################Applying ensemble Learning Algorithms-02####################################################


# In[60]:


#Decision Tree Algorithm 
#Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
DecisionTree_Model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0).fit(x_train, y_train)


# In[61]:


##### MEASUING THE PERFORMANCE OF THE DECISION TREE ALGORITHM ##################################


# In[62]:


predictedDT = DecisionTree_Model.predict(x_test)

score_DT = metrics.accuracy_score(y_test, predictedDT)
print(f'Accuracy: {round(score_DT*100,2)}%')

precision = precision_score(y_test,predictedDT)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predictedDT)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predictedDT)
print(f'f1_score: {round(f1*100,2)}%')


cm = metrics.confusion_matrix(y_test, predictedDT)
plot_confusion_matrix(cm, classes=['FAKE','REAL'])


# In[63]:


########################### CREATING THE AdaBoost-DECISION_TREE CLASSIFICATION ################################


# In[103]:


############Decision Tree Model-01(Experiment-1)

DecisionTree_Model1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Creating AdaBoost Model
Ada_DecisionTree_Model1 = AdaBoostClassifier(n_estimators=220,base_estimator=DecisionTree_Model1, learning_rate = 1).fit(x_train, y_train)


# In[ ]:


################### Measuring the performance of AdaBoost-DecisionTree ######################################


# In[104]:


#Predict the response of AdaBoost-DecisionTree
predicted_AdaBoostDT = Ada_DecisionTree_Model1.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostDT))

score_DT1 = metrics.accuracy_score(y_test, predicted_AdaBoostDT)
print(f'Accuracy: {round(score_DT*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostDT)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostDT)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostDT)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostDT)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[105]:


##########Decision Tree -02 (Experiment-2)
DecisionTree_Model2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Creating AdaBoost Model
Ada_DecisionTree_Model2 = AdaBoostClassifier(n_estimators=100,base_estimator=DecisionTree_Model2, learning_rate = 1).fit(x_train, y_train)

################### Measuring the performance of AdaBoost-DecisionTree ######################################

#Predict the response of AdaBoost-DecisionTree
predicted_AdaBoostDT2 = Ada_DecisionTree_Model2.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostDT2))

score_DT2 = metrics.accuracy_score(y_test, predicted_AdaBoostDT2)
print(f'Accuracy: {round(score_DT2*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostDT2)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostDT2)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostDT2)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostDT2)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[106]:


##########DT-03 (Experiment-3)
DecisionTree_Model3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Creating AdaBoost Model
Ada_DecisionTree_Model3 = AdaBoostClassifier(n_estimators=150,base_estimator=DecisionTree_Model3, learning_rate = 1).fit(x_train, y_train)

################### Measuring the performance of AdaBoost-DecisionTree ######################################

#Predict the response of AdaBoost-DecisionTree
predicted_AdaBoostDT3 = Ada_DecisionTree_Model3.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostDT3))

score_DT3 = metrics.accuracy_score(y_test, predicted_AdaBoostDT3)
print(f'Accuracy: {round(score_DT3*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostDT3)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostDT3)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostDT3)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostDT3)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[86]:


##########DT-04 (Experiment-4)
DecisionTree_Model4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Creating AdaBoost Model
Ada_DecisionTree_Model4 = AdaBoostClassifier(n_estimators=200,base_estimator=DecisionTree_Model4, learning_rate = 1).fit(x_train, y_train)

################### Measuring the performance of AdaBoost-DecisionTree ######################################

#Predict the response of AdaBoost-DecisionTree
predicted_AdaBoostDT4 = Ada_DecisionTree_Model4.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostDT4))

score_DT4 = metrics.accuracy_score(y_test, predicted_AdaBoostDT4)
print(f'Accuracy: {round(score_DT4*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostDT4)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostDT4)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostDT4)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostDT4)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[87]:


###########################DT-05 (Experiment-5)

DecisionTree_Model5 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Creating AdaBoost Model
Ada_DecisionTree_Model5 = AdaBoostClassifier(n_estimators=250,base_estimator=DecisionTree_Model5, learning_rate = 1).fit(x_train, y_train)

################### Measuring the performance of AdaBoost-DecisionTree ######################################

#Predict the response of AdaBoost-DecisionTree
predicted_AdaBoostDT5 = Ada_DecisionTree_Model5.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostDT5))

score_DT5 = metrics.accuracy_score(y_test, predicted_AdaBoostDT5)
print(f'Accuracy: {round(score_DT5*100,2)}%')

precision = precision_score(y_test, predicted_AdaBoostDT5)
print(f'precision: {round(precision*100,2)}%')

recall = recall_score(y_test, predicted_AdaBoostDT5)
print(f'recall: {round(recall*100,2)}%' )

f1 = f1_score(y_test, predicted_AdaBoostDT5)
print(f'f1_score: {round(f1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostDT5)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[107]:


## Comparing the results of various experiments on DecisionTree  Model
ACCURACYDT = np.vstack((score_DT,score_DT2,score_DT3,score_DT4,score_DT1,score_DT5))
model_numberDT = np.array([1,2,3,4,5,6])
print('Model number Represents:')
print(' 1 = DecisionTree')
print(' 2 = DecisionTree with n_estimator-100')
print(' 3 = DecisionTree with n_estimator-150')
print(' 4 = DecisionTree with n_estimator-200')
print(' 5 = DecisionTree with n_estimator-220')
print(' 6 = DecisionTree with n_estimator-250')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model_numberDT,ACCURACYDT, color = 'r', marker = 'o', linewidth = 5)
for j in range(0,len(ACCURACYDT)):
              ax.annotate('%3f'%(ACCURACYDT[j]), (model_numberDT[j],ACCURACYDT[j]))
plt.xlabel('MODELS')
plt.ylabel('ACCURACY')


# In[ ]:





# In[ ]:


###################################Applying ensemble Learning Algorithms-03####################################################


# In[37]:


# Implementing RandomForest
from sklearn.ensemble import RandomForestClassifier
RandomForest_Model = RandomForestClassifier().fit(x_train,y_train)

predictedRandomForest = RandomForest_Model.predict(x_test)

score_RF = metrics.accuracy_score(y_test, predictedRandomForest )
print(f'Accuracy: {round(score_RF*100,2)}%')

precisionRF = precision_score(y_test,predictedRandomForest )
print(f'precision: {round(precisionRF*100,2)}%')

recallRF = recall_score(y_test, predictedRandomForest )
print(f'recall: {round(recallRF*100,2)}%' )

f1_RF = f1_score(y_test, predictedRandomForest )
print(f'f1_score: {round(f1_RF*100,2)}%')


cm = metrics.confusion_matrix(y_test, predictedRandomForest )
plot_confusion_matrix(cm, classes=['FAKE','REAL'])


# In[ ]:


########################### CREATING THE AdaBoost-RandomForest CLASSIFICATION ################################


# In[39]:


#ADABoost classifier
#importing essential Libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings 


# In[40]:


#############################################RF-MODEL - 01 (Experiment-1)
from sklearn.ensemble import RandomForestClassifier
RF_Model1 = RandomForestClassifier()

Ada_RandomForest_Model1 = AdaBoostClassifier(n_estimators=100,base_estimator=RF_Model1, learning_rate = 1).fit(x_train, y_train)

predicted_AdaBoostRF1 = Ada_RandomForest_Model1.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostRF1))

score_RF1 = metrics.accuracy_score(y_test, predicted_AdaBoostRF1)
print(f'Accuracy: {round(score_RF1*100,2)}%')

precisionRF1 = precision_score(y_test, predicted_AdaBoostRF1)
print(f'precision: {round(precisionRF1*100,2)}%')

recallRF1 = recall_score(y_test, predicted_AdaBoostRF1)
print(f'recall: {round(recallRF1*100,2)}%' )

f1_RF1 = f1_score(y_test, predicted_AdaBoostRF1)
print(f'f1_score: {round(f1_RF1*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostRF1)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[41]:


#############################################RF-MODEL - 02 (Experiment-2)
from sklearn.ensemble import RandomForestClassifier
RF_Model2 = RandomForestClassifier()

Ada_RandomForest_Model2 = AdaBoostClassifier(n_estimators=150,base_estimator=RF_Model2, learning_rate = 1).fit(x_train, y_train)

predicted_AdaBoostRF2 = Ada_RandomForest_Model2.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostRF2))

score_RF2 = metrics.accuracy_score(y_test, predicted_AdaBoostRF2)
print(f'Accuracy: {round(score_RF2*100,2)}%')

precisionRF2 = precision_score(y_test, predicted_AdaBoostRF2)
print(f'precision: {round(precisionRF2*100,2)}%')

recallRF2 = recall_score(y_test, predicted_AdaBoostRF2)
print(f'recall: {round(recallRF2*100,2)}%' )

f1_RF2 = f1_score(y_test, predicted_AdaBoostRF2)
print(f'f1_score: {round(f1_RF2*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostRF2)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[42]:


#############################################RF-MODEL - 03 (Experiment-3)
from sklearn.ensemble import RandomForestClassifier
RF_Model3 = RandomForestClassifier()

Ada_RandomForest_Model3 = AdaBoostClassifier(n_estimators=200,base_estimator=RF_Model3, learning_rate = 1).fit(x_train, y_train)

predicted_AdaBoostRF3 = Ada_RandomForest_Model3.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostRF3))

score_RF3 = metrics.accuracy_score(y_test, predicted_AdaBoostRF3)
print(f'Accuracy: {round(score_RF3*100,2)}%')

precisionRF3 = precision_score(y_test, predicted_AdaBoostRF3)
print(f'precision: {round(precisionRF3*100,2)}%')

recallRF3 = recall_score(y_test, predicted_AdaBoostRF3)
print(f'recall: {round(recallRF2*100,2)}%' )

f1_RF3 = f1_score(y_test, predicted_AdaBoostRF3)
print(f'f1_score: {round(f1_RF3*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostRF3)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[43]:


#############################################RF-MODEL - 04 (Experiment-4)
from sklearn.ensemble import RandomForestClassifier
RF_Model4 = RandomForestClassifier()

Ada_RandomForest_Model4 = AdaBoostClassifier(n_estimators=220,base_estimator=RF_Model4, learning_rate = 1).fit(x_train, y_train)

predicted_AdaBoostRF4 = Ada_RandomForest_Model4.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostRF4))

score_RF4 = metrics.accuracy_score(y_test, predicted_AdaBoostRF4)
print(f'Accuracy: {round(score_RF4*100,2)}%')

precisionRF4 = precision_score(y_test, predicted_AdaBoostRF4)
print(f'precision: {round(precisionRF4*100,2)}%')

recallRF4 = recall_score(y_test, predicted_AdaBoostRF4)
print(f'recall: {round(recallRF4*100,2)}%' )

f1_RF4 = f1_score(y_test, predicted_AdaBoostRF4)
print(f'f1_score: {round(f1_RF4*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostRF4)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[44]:


#############################################RF-MODEL - 05 (Experiment-5)
from sklearn.ensemble import RandomForestClassifier
RF_Model5 = RandomForestClassifier()

Ada_RandomForest_Model5 = AdaBoostClassifier(n_estimators=250,base_estimator=RF_Model5, learning_rate = 1).fit(x_train, y_train)

predicted_AdaBoostRF5 = Ada_RandomForest_Model5.predict(x_test)
print("Accuracy :", metrics.accuracy_score(y_test, predicted_AdaBoostRF5))

score_RF5 = metrics.accuracy_score(y_test, predicted_AdaBoostRF5)
print(f'Accuracy: {round(score_RF5*100,2)}%')

precisionRF5 = precision_score(y_test, predicted_AdaBoostRF5)
print(f'precision: {round(precisionRF5*100,2)}%')

recallRF5 = recall_score(y_test, predicted_AdaBoostRF5)
print(f'recall: {round(recallRF5*100,2)}%' )

f1_RF5 = f1_score(y_test, predicted_AdaBoostRF5)
print(f'f1_score: {round(f1_RF5*100,2)}%')

cm = metrics.confusion_matrix(y_test, predicted_AdaBoostRF5)
plot_confusion_matrix(cm, classes=['FAKE Data','REAL Data'])


# In[45]:


## Comparing the results of various experiments on Naive bayes Model
ACCURACYRF = np.vstack((score_RF,score_RF1,score_RF2,score_RF3,score_RF4,score_RF5))
model_number = np.array([1,2,3,4,5,6])
print('Model number Represents:')
print(' 1 =  Random Forest Model')
print(' 2 = AdaBoost_Random Forest with n_estimator-100')
print(' 3 = AdaBoost_Random Forest with n_estimator-150')
print(' 4 = AdaBoost_Random Forest with n_estimator-200')
print(' 5 = AdaBoost_Random Forest with n_estimator-220')
print(' 6 = AdaBoost_Random Forest with n_estimator-250')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model_number,ACCURACYRF, color = 'b', marker = '*', linewidth = 5)
for j in range(0,len(ACCURACYRF)):
              ax.annotate('%3f'%(ACCURACYRF[j]), (model_number[j],ACCURACYRF[j]))
plt.xlabel('MODELS')
plt.ylabel('ACCURACY')


# In[68]:


### Accuacy graph of Experimented RandomForest models  
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')
print('Model number Represents:')
print(' 1 =  Random Forest Model')
print(' 2 = AdaBoost_Random Forest with n_estimator-100')
print(' 3 = AdaBoost_Random Forest with n_estimator-150')
print(' 4 = AdaBoost_Random Forest with n_estimator-200')
print(' 5 = AdaBoost_Random Forest with n_estimator-220')
print(' 6 = AdaBoost_Random Forest with n_estimator-250')


ACCURACY = np.vstack((score_RF,score_RF1,score_RF2,score_RF3,score_RF4,score_RF5))
models =np.array([1,2,3,4,5,6])
plt.title("output accuracy result")
style.use('ggplot')
plt.plot(models,ACCURACY,'r')
plt.xlabel('models')
plt.ylabel('Accuracy')
plt.show()


# In[87]:


### Accuacy graph of Experimented RandomForest models  
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')
print('Model number Represents:')
print(' 1 =  Random Forest Model')
print(' 2 = AdaBoost_Random Forest with n_estimator-100')
print(' 3 = AdaBoost_Random Forest with n_estimator-150')
print(' 4 = AdaBoost_Random Forest with n_estimator-200')
print(' 5 = AdaBoost_Random Forest with n_estimator-220')
print(' 6 = AdaBoost_Random Forest with n_estimator-250')


ACCURACY = np.vstack((score_RF,score_RF1,score_RF2,score_RF3,score_RF4,score_RF5))
models =np.array([1,2,3,4,5,6])
plt.title("output accuracy result")
style.use('ggplot')
plt.plot(models,ACCURACY,'r',label='Random Forest',linewidth=2)
plt.xlabel('models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[89]:


### Accuacy graph of Experimented DecisionTree models  
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')
print('Model number Represents:')
print(' 1 = DecisionTree')
print(' 2 = AdaBoost_DecisionTree with n_estimator-100')
print(' 3 = AdaBoost_DecisionTree with n_estimator-150')
print(' 4 = AdaBoost_DecisionTree with n_estimator-200')
print(' 5 = AdaBoost_DecisionTree with n_estimator-220')
print(' 6 = AdaBoost_DecisionTree with n_estimator-250')


ACCURACYDT = np.vstack((score_DT,score_DT2,score_DT3,score_DT4,score_DT1,score_DT5))
model_numberDT=np.array([1,2,3,4,5,6])

style.use('ggplot')
plt.plot(model_numberDT,ACCURACYDT,'b',label='Decision Tree',linewidth=2)
plt.title("Output Accuracy Result of Models")
plt.xlabel('models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[90]:


### Accuacy graph of Experimented Naive Bayes models  
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')
print('Model number Represents:')
print(' 1 = NaiveBayes')
print(' 2 = AdaBoost_NaiveBayes with n_estimator-100')
print(' 3 = AdaBoost_NaiveBayes with n_estimator-150')
print(' 4 = AdaBoost_NaiveBayes with n_estimator-200')
print(' 5 = AdaBoost_NaiveBayes with n_estimator-220')
print(' 6 = AdaBoost_NaiveBayes with n_estimator-250')


ACCURACYNB = np.vstack((score,score_ANB2,score_ANB3,score_ANB4,score_ANB,score_ANB5))
model =np.array([1,2,3,4,5,6])

style.use('ggplot')
plt.plot(model,ACCURACYNB,'g',label='Naive Bayes',linewidth=2)
plt.title("Output Accuracy Result of Models")
plt.xlabel('models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[93]:



### Comparison of various models
import matplotlib.pyplot as plt
from matplotlib import style

get_ipython().run_line_magic('matplotlib', 'inline')

ACCURACYNB = np.vstack((score,score_ANB2,score_ANB3,score_ANB4,score_ANB,score_ANB5))
ACCURACYDT = np.vstack((score_DT,score_DT2,score_DT3,score_DT4,score_DT1,score_DT5))
#ACCURACY = np.vstack((score_RF,score_RF1,score_RF2,score_RF3,score_RF4,score_RF5))
ACCURACY = np.vstack((a,b,c,d,e,f))
model =np.array([1,2,3,4,5,6])

style.use('ggplot')
plt.plot(model,ACCURACYNB,'r',label='Naive Bayes',linewidth=2)
plt.plot(model,ACCURACYDT,'g',label='Decision Tree',linewidth=2.5)
plt.plot(model ,ACCURACY,'b',label='Random Forest',linewidth=2)

plt.axis([0,7,86,99])
plt.title("Comparison of various models")
plt.xlabel('models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




