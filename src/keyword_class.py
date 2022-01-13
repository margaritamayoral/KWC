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
###############################
#### Grid search for parameter tunning ###
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import GridSearchCV

################################
### Deep Learning model  ###
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
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

#df = pd.read_json('../../keyword_categories/keyword_categories_train.jsonl', lines=True)
#df_test = pd.read_json('../../keyword_categories/keyword_categories_test.jsonl', lines=True)
#
####### Data training exploration ##################
##looking the training data
#print(df.head(10))
##looking for missings, kind of data and shape:
#print(df.info())
####### Data Cleaning and Normalization ####
#print(df['categories'].head(10))
#df = df.explode('categories')
#print(df['categories'].head(10))
#print(df.head(10))
#print(df.info())
#
#keywords = df['keyword']
#print("detecting languages in training data")
#languages_langdetect = []
#for line in keywords:
#    try:
#        result = langdetect.detect_langs(line)
#        #print(result)
#        result = str(result[0])[:2]
#        #print(result)
#    except:
#        result = 'unknown'
#
#    finally:
#        languages_langdetect.append(result)
#df['languages_langdetect'] = languages_langdetect
#print(df.head(10))
### dropping the rows which contains languages that are not english  ###
#df.drop(df[df['languages_langdetect'] != 'en'].index, inplace = True)
#print(df.head(10))
#print(df.info())
##
### saving in csv ##
#df.to_csv('../data/keyword_categories_train_clean.csv')
##
##########  Data test exploration ###########
#print(df_test.head(10))
#print(df_test.info())
#df_test = df_test.explode('categories')
#print(df_test['categories'].head(10))
#print(df_test.head())
#print(df_test.info())
#keywords_test = df_test['keyword']
#print("detecting languages 2")
#languages_langdetect_test = []
#for line in keywords_test:
#    try:
#        result = langdetect.detect_langs(line)
#        #print(result)
#        result = str(result[0])[:2]
#        #print(result)
#    except:
#        result = 'unknown'
#
#    finally:
#        languages_langdetect_test.append(result)
#df_test['languages_langdetect'] = languages_langdetect_test
#print(df_test.head(10))
### dropping the rows which contains languages that are not english  ###
#df_test.drop(df_test[df_test['languages_langdetect'] != 'en'].index, inplace = True)
#print(df_test.head(10))
#print(df_test.info())
### saving in csv ##
#df.to_csv('../data/keyword_categories_test_clean.csv')
#
#
#
## reading the new csvs ##
df = pd.read_csv('../data/keyword_categories_train_clean.csv')
print(df.head(10))
df_test = pd.read_csv('../data/keyword_categories_test_clean.csv')
print(df_test.head(10))

counter = Counter(df['categories'].tolist())
print(counter)
#
top_10_categories = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['categories'].map(lambda x: x in top_10_categories)]
print(df.head())
keyword_list = df['keyword'].tolist()
categories_list = [top_10_categories[i] for i in df['categories'].tolist()]
print(categories_list)
categories_list = np.array(categories_list)
print(categories_list)

count_vect = CountVectorizer(stop_words='english')
x_train_counts = count_vect.fit_transform(keyword_list)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

## test data ##
counter_test = Counter(df_test['categories'].tolist())
print(counter_test)
#top_10_categories_test = {i[0]: idx for idx, i in enumerate(counter_test.most_common(10))}
df_test = df_test[df_test['categories'].map(lambda x: x in top_10_categories)]
print(df_test.head())
keyword_list_test = df_test['keyword'].tolist()
categories_list_test = [top_10_categories[i] for i in df_test['categories'].tolist()]
print(categories_list_test)
categories_list_test = np.array(categories_list_test)
print(categories_list_test)
x_test_counts = count_vect.fit_transform(keyword_list_test)
x_test_tfidf = tfidf_transformer.fit_transform(x_test_counts)


train_x = x_train_tfidf
test_x = x_test_tfidf
train_y = categories_list
test_y = categories_list_test
#train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, categories_list, test_size=0.3)
#
##########################################
##### Naive Bayes Classifier  ####
clf = MultinomialNB().fit(train_x, train_y)
print(clf)
#
y_score = clf.predict(test_x)
print(test_x, y_score)
print("the NB y score is:", y_score)
#
#text_clf = Pipeline([
#('vect', CountVectorizer(stop_words='english')),
#('tfidf', TfidfTransformer()),
#('clf', MultinomialNB()),
#])
##



#
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
#
##
###### It gave an accuracy of about  36.43%###
#
######################################
##### Support Vector Machines (SVM) ###
##
print("computing SGD Classifier")
clf_svm_2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42).fit(train_x, train_y)
y_score_svm_2 = clf_svm_2.predict(test_x)
print(y_score_svm_2)
##
n_right_svm_2 = 0
for i in range(len(y_score_svm_2)):
    if y_score_svm_2[i] == test_y[i]:
        n_right_svm_2 += 1
##
print("SGD Classifier Accuracy: %.2f%%" % ((n_right_svm_2/float(len(test_y)) * 100)))

###### It gave an accuracy of about  38.71%###
#
### Deep learning model  ###
print("Calculating Deep Learning model")
mapped_list, word_list = filter_to_top_x(keyword_list, 2500, 10)

categories_list_o = [top_10_categories[i] for i in df['categories'].tolist()]
print(categories_list_o)
categories_list = to_categorical(categories_list_o)
#
max_keyword_length = 150
#
mapped_list = sequence.pad_sequences(mapped_list, maxlen=max_keyword_length)
#train_x, test_x, train_y, test_y = train_test_split(mapped_list, categories_list, test_size=0.3)
#
max_keyword_length = 150
embedding_vector_length = 64
print("training DL model ...")
model = Sequential()
#
model.add(Embedding(2500, embedding_vector_length, input_length=max_keyword_length))
model.add(Conv1D(50,5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(max(categories_list_o) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=3, batch_size=64)
#
y_score = model.predict(test_x)
y_score = [[1 if i == max(sc) else 0 for i in sc] for sc in y_score]
n_right = 0
for i in range(len(y_score)):
    if all(y_score[i][j] == test_y[i][j] for j in range(len(y_score[i]))):
        n_right += 1
print("Accuracy DL: %.2f%%" % ((n_right/float(len(test_y)) * 100)))


