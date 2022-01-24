from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import json
#import keras
#import seaborn as sns
#import matplotlib as plt
#import matplotlib.pyplot as plt
#from scipy.stats import norm
from sklearn.pipeline import Pipeline
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split

#################################
### NLP libraries  ######
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from spacy.tokens import Token


from langdetect import detect

import langdetect
#import langid

from textblob import TextBlob

from googletrans import Translator
#import nltk
#from nltk.stem.snowball import SnowballStemmer

### CLASSIFIERS  ###
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
###############################
#### Grid search for parameter tunning ###
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import GridSearchCV

################################
### Deep Learning model  ###
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
#
#################################################
### some functions  ####
def count_top_x_words(corpus, top_x, skip_top_n):
    count = defaultdict(lambda: 0)
    for c in corpus:
        for w in word_tokenize(c):
            count[w] +=1
    count_tuples = sorted([(w, c) for w, c in count.items()], key=lambda x: x[1], reverse=True)
    return [i[0] for i in count_tuples[skip_top_n: skip_top_n + top_x]]
    
    
def replace_top_x_words_with_vectors(corpus, top_x):
    topx_dict = {top_x[i]: i for i in range(len(top_x))}
    
    return [
        [topx_dict[w] for w in word_tokenize(s) if w in topx_dict]
        for s in corpus
    ], topx_dict


def filter_to_top_x(corpus, n_top, skip_n_top=0):
    top_x = count_top_x_words(corpus, n_top, skip_n_top)
    return replace_top_x_words_with_vectors(corpus, top_x)
    
def get_lang_detector(nlp, name):
    return LanguageDetector()

###################################################
### Loading the language of use ###
nlp = spacy.load('en_core_web_sm')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)


#################################################

##### Loading the data ####

df = pd.read_json('../../keyword_categories/keyword_categories_train.jsonl', lines=True)
df_test = pd.read_json('../../keyword_categories/keyword_categories_test.jsonl', lines=True)
#
####### Data training exploration ##################
##looking the training data
#print(df.head(10))
##looking for missings, kind of data and shape:
#print(df.info())
######## Data Cleaning and Normalization ####
#print(df['categories'].head(10))
df = df.explode('categories')
#print(df['categories'].head(10))
#print(df.head(10))
#print(df.info())
##
keywords = df['keyword']
print("detecting languages in training data")
languages_langdetect = []
for line in keywords:
    try:
        result = langdetect.detect_langs(line)
#        #print(result)
        result = str(result[0])[:2]
#        #print(result)
    except:
        result = 'unknown'
#
    finally:
        languages_langdetect.append(result)
df['languages_langdetect'] = languages_langdetect
#print(df.head(10))
#### dropping the rows which contains languages that are not english  ###
df.drop(df[df['languages_langdetect'] != 'en'].index, inplace = True)
#print(df.head(10))
#print(df.info())
###
#### saving in csv ##
df.to_csv('../data/keyword_categories_train_clean.csv')
###
###########  Data test exploration ###########
#print(df_test.head(10))
#print(df_test.info())
df_test = df_test.explode('categories')
#print(df_test['categories'].head(10))
#print(df_test.head())
#print(df_test.info())
keywords_test = df_test['keyword']
print("detecting languages 2")
languages_langdetect_test = []
for line in keywords_test:
    try:
        result = langdetect.detect_langs(line)
        #print(result)
        result = str(result[0])[:2]
        #print(result)
    except:
        result = 'unknown'

    finally:
        languages_langdetect_test.append(result)
df_test['languages_langdetect'] = languages_langdetect_test
#print(df_test.head(10))
#### dropping the rows which contains languages that are not english  ###
df_test.drop(df_test[df_test['languages_langdetect'] != 'en'].index, inplace = True)
#print(df_test.head(10))
#print(df_test.info())
#### saving in csv ##
df.to_csv('../data/keyword_categories_test_clean.csv')
#
#
#
## reading the new csvs ##
df = pd.read_csv('../data/keyword_categories_train_clean.csv')
#print(df.head(10))
#print(df.info())
#print(df['keyword'].head(20))
#print(df['categories'].head(20))
df_test = pd.read_csv('../data/keyword_categories_test_clean.csv')
#print(df_test.head(10))


#### Counting the different categories and taking only the 10 top (most common)#####

counter = Counter(df['categories'].tolist())
#print(counter)
#
top_10_categories = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
#print(top_10_categories)
df = df[df['categories'].map(lambda x: x in top_10_categories)]
#print(df.head(5))
#print(df['categories'].head(5))
#print(df['keyword'].head(5))
#print(df.info())
keyword_list = df['keyword'].tolist()
#print(keyword_list)
categories_list = [top_10_categories[i] for i in df['categories'].tolist()]
#print(categories_list)
categories_list = np.array(categories_list)
#print(categories_list)
#### Test data  #####
df_test = df_test[df_test['categories'].map(lambda x: x in top_10_categories)]
keyword_list_test = df_test['keyword'].tolist()
categories_list_test = [top_10_categories[i] for i in df_test['categories'].tolist()]
categories_list_test = np.array(categories_list_test)
#####################################################
##### vectorizing the data  Method A ####
### Training data  ###
#print(df.head())
count_vect = CountVectorizer(stop_words='english')
count_vect = count_vect.fit(keyword_list)
tfidf_transformer = TfidfTransformer()
x_train = count_vect.transform(keyword_list)
print("x_train", x_train)
y_train = categories_list
### Test data  ##
x_test = count_vect.transform(keyword_list_test)
print("x_test", x_test)
y_test = categories_list_test

####### Logistic Regression model #####
print("Training Logistic Regression model")
LRmodel = LogisticRegression()
LRmodel.fit(x_train, y_train)
print("Testing Logistic Regression model")
score = LRmodel.score(x_test, y_test)
print("LR Accuracy: ", score)
###### It gave an accuracy of about  46.15%###

########################################################
#### Vectorizing the data Method B  ################

count_vect = CountVectorizer(stop_words='english')
x_train_counts = count_vect.fit_transform(keyword_list)
#print("printing the vector of counts", x_train_counts)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

## test data ##
df_test = df_test[df_test['categories'].map(lambda x: x in top_10_categories)]
#print(df_test.head())
keyword_list_test = df_test['keyword'].tolist()
categories_list_test = [top_10_categories[i] for i in df_test['categories'].tolist()]
#print(categories_list_test)
categories_list_test = np.array(categories_list_test)
#print(categories_list_test)
x_test_counts = count_vect.fit_transform(keyword_list_test)
x_test_tfidf = tfidf_transformer.fit_transform(x_test_counts)


train_x = x_train_tfidf
test_x = x_test_tfidf
train_y = categories_list
test_y = categories_list_test

#
##########################################
##### Naive Bayes Classifier  ####
print("Training Naive Bayes Classifier")
clf = MultinomialNB().fit(train_x, train_y)
#print(clf)
#
print("Testing Naive Bayes Classifier")
y_score = clf.predict(test_x)
#print(test_x, y_score)
#
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("The NB Accuracy is: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
#
##
###### It gave an accuracy of about  46.26%###
#
######################################
##### Support Vector Machines (SVM) ###
##
print("computing SGD Classifier")
clf_svm_2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42).fit(train_x, train_y)
print("testing SGC Classifier")
y_score_svm_2 = clf_svm_2.predict(test_x)
#print(y_score_svm_2)
##
n_right_svm_2 = 0
for i in range(len(y_score_svm_2)):
    if y_score_svm_2[i] == test_y[i]:
        n_right_svm_2 += 1
##
print("SGD Classifier Accuracy: %.2f%%" % ((n_right_svm_2/float(len(test_y)) * 100)))

###### It gave an accuracy of about 47.42%###
#
### Deep learning model  ###
print("Calculating Deep Learning model")

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(keyword_list)
Xcnn_train = tokenizer.texts_to_sequences(keyword_list)
Xcnn_test = tokenizer.texts_to_sequences(keyword_list_test)
vocab_size = len(tokenizer.word_index) + 1
print(keyword_list[1])
print(Xcnn_train[1])
maxlen = 100
Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
Xcnn_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)
print(Xcnn_train[0, :])
##### adding layers  ####
embedding_dim = 200
textcnnmodel = Sequential()
textcnnmodel.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
textcnnmodel.add(layers.Conv1D(128, 5, activation='relu'))
textcnnmodel.add(layers.GlobalMaxPooling1D())
textcnnmodel.add(layers.Dense(10,activation='relu'))
textcnnmodel.add(layers.Dense(1,activation='sigmoid'))
textcnnmodel.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
textcnnmodel.summary()
###### Fitting the model  ###
textcnnmodel.fit(Xcnn_train, y_train, epochs=4, validation_data=(Xcnn_test, y_test), batch_size=10)
loss, accuracy = textcnnmodel.evaluate(Xcnn_train, y_train, verbose=False)
print("training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = textcnnmodel.evaluate(Xcnn_test, y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))



