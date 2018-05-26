#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Apr 28 10:06:35 2018

@author: dineshkaimal91
"""

from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Text Topic Imports
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from Class_tree import DecisionTree
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import random

def my_analyzer(s):
    # Synonym List
    syns = {'veh': 'vehicle', 'car': 'vehicle', 'hond':'honda', \
              'tl':'till', 'air bag': 'airbag', \
              'seat belt':'seatbelt', "n't":'not', 'to30':'to 30', \
              'wont':'would not', 'cant':'can not', 'cannot':'can not', \
              'couldnt':'could not', 'shouldnt':'should not', \
              'wouldnt':'would not', 'straightforward': 'straight forward',\
              'takada':'takata'}
    
    
    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    # Tokenize
    tokens = word_tokenize(s)
    #tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and \
              ("''" != word) and ("``" != word) and \
              (word!='description') and (word !='dtype') \
              and (word != 'object') and (word!="'s")]
    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]
    # Remove stop words
    others = ["quot", "say", "could", "also", "even", "really", "one", \
    "would", "get", "getting", "go", "going", "..", "...", \
    "us", "area", "need","oct", "place", "want", "get", \
    "take", "end", "come", "gal", "get", "next", "though", \
    "say", "seem", "use", "sep", "w/", "jul"]
    stop = stopwords.words('english') + others
    
    tokens = [word for word in tokens if (word not in stop)]
    filtered_terms = [word for word in tokens if (len(word)>1) and \
                      (not word.replace('.','',1).isnumeric()) and \
                      (not word.replace("'",'',2).isnumeric())]
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos = tagged_token[1]
        pos = pos[0]
        try:
            pos = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens
    



def my_preprocessor(s):
    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    return(s)


def display_topics(topic_vectorizer, terms, n_terms=15, word_cloud=False, mask=None):
    for topic_idx, topic in enumerate(topic_vectorizer):
        message = "Topic #%d: " %(topic_idx+1)
        print(message)
        abs_topic = abs(topic)
        topic_terms_sorted = \
            [[terms[i], topic[i]] \
                 for i in abs_topic.argsort()[:-n_terms - 1:-1]]
        k = 5
        n = int(n_terms/k)
        m = n_terms - k*n
        for j in range(n):
            l = k*j
            message = ''
            for i in range(k):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        if m> 0:
            l = k*n
            message = ''
            for i in range(m):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        print("")
      
    return
def term_dic(tf, terms, scores=None):
    td = {}
    for i in range(tf.shape[0]):
    # Iterate over the terms with nonzero scores
        term_list = tf[i].nonzero()[1]
        if len(term_list)>0:
            if scores==None:
                for t in np.nditer(term_list):
                    if td.get(terms[t]) == None:
                        td[terms[t]] = tf[i,t]
                    else:
                        td[terms[t]] += tf[i,t]
        else:
            for t in np.nditer(term_list):
                score = scores.get(terms[t])
                if score != None:
                    # Found Sentiment Word
                    score_weight = abs(scores[terms[t]])
                    if td.get(terms[t]) == None:
                        td[terms[t]] = tf[i,t] * score_weight
                    else:
                        td[terms[t]] += tf[i,t] * score_weight
    return td

# Increase column width to let pandy read large text columns
pd.set_option('max_colwidth', 32000)
# Read NHTSA Comments
file_path = r'C:/Users/siddh/Desktop/EM_Projects/Finals/Python'
df = pd.read_excel(file_path+"/HondaComplaints.xlsx")


''''
*********Data exploration and visualization*************
sns.factorplot(x="crash",y='mph',col="Year",data=df,kind="box",size=4,aspect=.7)
sns.factorplot(x="crash",y='mph',data=df,kind="box",size=4,aspect=.7)
sns.factorplot(x="crash",y='mileage',col="Year",data=df,kind="violin",size=4,aspect=.7)
'''
# Setup program constants and reviews
n_reviews = len(df['description'])
n_topics = 7 # number of topics

# Create Word Frequency by Review Matrix using Custom Analyzer
# max_df is a stop limit for terms that have more than this
# proportion of documents with the term (max_df - don't ignore any terms)
cv = CountVectorizer(max_df=0.7, min_df=4, max_features=None,\
                     analyzer=my_analyzer)
tf = cv.fit_transform(df['description'])
tf1=tf
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))
# Term Dictionary with Terms as Keys and frequency as Values
td = term_dic(tf, terms)
print("The Corpus contains a total of ", len(td), " unique terms.")
print("The total number of terms in the Corpus is", sum(td.values()))

term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))


# Construct the TF/IDF matrix from the data
print("\nConducting Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_features
tf = tfidf_vect.fit_transform(tf)
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",\
      tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0], \
          term_idf_scores[j][1]))



# LDA Analysis
uv = LatentDirichletAllocation(n_components=n_topics, \
                               learning_method='online', random_state=12345)
U = uv.fit_transform(tf)
print("\n********** GENERATED TOPICS **********")
display_topics(uv.components_, terms, n_terms=15)
# Store topic selection for each doc in topics[]
topics = [0] * n_reviews
for i in range(n_reviews):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
U_rev_scores = []
for i in range(n_reviews):
    u = [0] * (n_topics+1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j+1] = U[i][j]
    U_rev_scores.append(u)
rev_scores = U_rev_scores
# Integrate Topic Scores into Main Data Frame (df)
cols = ["topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols)
df = df.join(df_rev)


print(" TOPIC DISTRIBUTION")
print('{:<6s}{:>4s}{:>12s}'.format("TOPIC", "N", "PERCENT"))
print("----------------------")
topic_counts = df['topic'].value_counts(sort=False)
for i in range(len(topic_counts)):
    percent = 100*topic_counts[i]/n_reviews
    print('{:>3d}{:>8d}{:>9.1f}%'.format((i+1), topic_counts[i], percent))


print("\n**** Sentiment Analysis ****")
sw = pd.read_excel(file_path+"/afinn_sentiment_words.xlsx")
# Setup Sentiment dictionary
sentiment_dic = {}
for i in range(len(sw)):
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]
# Display first ten terms in the dictionary
n = 0
for k,v in sentiment_dic.items():
    n += 1
    print(k, v)
    if n>10:
        break


tf=tf1

# Calculate average sentiment for every review save in sentiment_score[]
min_sentiment = +5
max_sentiment = -5
avg_sentiment, min, max = 0,0,0
min_list, max_list = [],[]
sentiment_score = [0]*n_reviews
for i in range(n_reviews):
    # Iterate over the terms with nonzero scores
    n_sw = 0
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score = sentiment_dic.get(terms[t])
            if score != None:
                sentiment_score[i] += score * tf[i,t]
                n_sw += tf[i,t]
    if n_sw>0:
        sentiment_score[i] = sentiment_score[i]/n_sw
    if sentiment_score[i]==max_sentiment and n_sw>3:
        max_list.append(i)
    if sentiment_score[i]>max_sentiment and n_sw>3:
        max_sentiment=sentiment_score[i]
        max = i
        max_list = [i]
    if sentiment_score[i]==min_sentiment and n_sw>3:
        min_list.append(i)
    
    if sentiment_score[i]<min_sentiment and n_sw>3:
        min_sentiment=sentiment_score[i]
        min = i
        min_list = [i]
    avg_sentiment += sentiment_score[i]
avg_sentiment = avg_sentiment/n_reviews
print("\nCorpus Average Sentiment:{:>5.2f} ".format(avg_sentiment))
print("\nMost Negative Reviews with 4 or more Sentiment Words:")
for i in range(len(min_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", min_list[i], \
          " Sentiment is ", min_sentiment))

print("\nMost Positive Reviews with 4 or more Sentiment Words:")
for i in range(len(max_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", max_list[i], \
          " Sentiment is ", max_sentiment))


df_sentiment = pd.DataFrame(sentiment_score, columns=['sentiment'])
df = df.join(df_sentiment)
print(df.groupby(['Model'])['sentiment'].mean(), "\n")
print(df.groupby(['topic'])['sentiment'].mean(), "\n")
print(df.groupby(['topic','Model'])['sentiment'].mean())



'''
*****States analysis code*****************

states = pd.read_csv("C:/Dinesh/TAMU/Courses/Spring 2018/Final/states.csv")
df_1 = df.merge(states, left_on = 'State', right_on = 'Abbreviation')
xy = df_1.groupby(["State_y","crash"]).size()
df_2 = xy.groupby(level = 0).apply(lambda x:100*x/float(x.sum())).reset_index(name = "fraction")
df_2 = df_2.loc[(df_2.crash == "Y")].sort_values(['fraction'])

xc = df_1.groupby(["State_y"]).size().reset_index(name = "Freq")
xc = xc.assign(p = 100*xc.Freq/xc.Freq.sum())
#xyd = xc.merge(states, left_on = 'State_y', right_on = 'State')
xyd = xc.merge(df_2, left_on = 'State_y', right_on = 'State_y')
xyd = xyd.assign(score = xyd.p*xyd.fraction).sort_values(['score'],ascending = False)



'''
 
print(" | Sentiment Terms | Average |")
print("Topic | Unique Total | Sentiment |")
print("------------------------------------")
topic_avg_sentiment = df.groupby(['topic'])['sentiment'].mean()
sentiment_clouds=[] # List of cloud dictionaries
for i in range(n_topics):
    idx = df.index[df['topic']==i] # Index List for Topic i
    topic_sentiment = {}
    n = 0
    # Iterate over the topic i sentiments
    for j in idx:
        # Iterate over the terms with nonzero scores
        term_list = tf[j].nonzero()[1]
        if len(term_list)>0:
            for t in np.nditer(term_list):
                score = sentiment_dic.get(terms[t])
                if score != None:
                    # Found Sentiment Word
                    n += tf[j,t]
                    score_weight = abs(sentiment_dic[terms[t]])
                    current_count = topic_sentiment.get(terms[t])
                    if current_count == None:
                        topic_sentiment[terms[t]] = tf[j,t]*score_weight
                    else:
                        topic_sentiment[terms[t]]+= tf[j,t]*score_weight
sentiment_clouds.append(topic_sentiment)
print("{:>3d}{:>10d}{:>10d}{:>9.2f}".format((i+1), len(topic_sentiment), \
      n, topic_avg_sentiment[i]))    




attribute_map = {
    'NhtsaID':[3,(0, 1e+12),[0,0]],
    'Year':[2,(2001, 2002, 2003),[0,0]],
    'Make':[2,('HONDA','ACURA'),[0,0]],
    'Model':[2,('TL','ODYSSEY','CR-V','CL','CIVIC','ACCORD'),[0,0]],
    'description':[3,(''),[0,0]],
    'State':[3,(''),[0,0]],
    'crash':[1,('N', 'Y'),[0,0]],
    'abs':[1,('N', 'Y'),[0,0]],
    'cruise':[1,('N', 'Y'),[0,0]],
    'mileage':[0,(0, 200000),[0,0]],
    'mph':[0,(0,80),[0,0]],
    'topic':[2,(0,1,2,3,4,5,6),[0,0]],
    'sentiment':[0,(-5,5),[0,0]],
    'T1':[0,(-1e+8,1e+8),[0,0]],
    'T2':[0,(-1e+8,1e+8),[0,0]],
    'T3':[0,(-1e+8,1e+8),[0,0]],
    'T4':[0,(-1e+8,1e+8),[0,0]],
    'T5':[0,(-1e+8,1e+8),[0,0]],
    'T6':[0,(-1e+8,1e+8),[0,0]],
    'T7':[0,(-1e+8,1e+8),[0,0]]}


target   = 'crash'
# Drop data with missing values for target 
drops= []
for i in range(df.shape[0]):
    if pd.isnull(df['crash'][i]):
        drops.append(i)
df = df.drop(drops)
df = df.reset_index()

encoding = 'one-hot' 
scale    = None # Interval scaling:  Use 'std', 'robust' or None
# drop=False - do not drop last category - used for Decision Trees
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding=encoding, \
                          interval_scale = scale, drop=False, display=True)
encoded_df = rie.fit_transform(df)
varlist = [target, 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
X = encoded_df.drop(varlist, axis=1)
y = encoded_df[target]
np_y = np.ravel(y) #convert dataframe column to flat array
col  = rie.col
for i in range(len(varlist)):
    col.remove(varlist[i])


maxdep_list=[4,5,6,7,8,9]
score_list = ['accuracy', 'recall', 'precision', 'f1']

for maxdep in maxdep_list:
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=maxdep)
    dtc = dtc.fit(X,y)
    mean_score = []
    std_score = []
    print("Max depth : %s" % maxdep)
    #print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
        mean = dtc_10.mean()
        std = dtc_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                
print("Splitting the dataset and comparing Training and Validation:")
X_train, X_validate, y_train, y_validate = \
            train_test_split(X,y,test_size = 0.3, random_state=7)
#Fitting a tree with the best depth using the training data
dtc1 = DecisionTreeClassifier(criterion='entropy', max_depth=9)
dtc1 = dtc1.fit(X_train,y_train)
features = list(X)
DecisionTree.display_importance(dtc1, features)
DecisionTree.display_binary_metrics(dtc1, X_validate, y_validate)






#df_senti=pd.DataFrame(sentiment_score)
#df2=df.join(df_senti)
print("\nAverageSentiment by Cluster:")
#df2.columns=['url', 'Article', 'topic', 'T1', 'T2', 'T3', 'T4', 'T5', 'senti']
df2=df.drop(['NhtsaID','T1','T2','T3','T4','T5','T6','mileage','T7','mph','Year',\
             'index'],axis=1)
df3=df2.groupby(['topic']).mean()
print(df3)


## WORD CLOUDS
print("***************************************************************")
print("WordClouds........")


stopw = set(STOPWORDS)

def shades_of_grey(word, font_size, position, orientation, random_state=None, \
**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)
#circle_mask = np.array(Image.open("CircleMask.png"))

## ALL WORDS IN REVIEW - ALL SENTIMENT WORDS IN REVIEW:
    
wrd=" ".join(pd.Series(df2['description']).astype(str))

def shades_of_grey(word, font_size, position, orientation, random_state=None, \
**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)

wcm1 = WordCloud(background_color="grey", max_words=200, stopwords=stopw, collocations=False,  \
random_state=341)
wcm1.generate(wrd)

plt.imshow(wcm1.recolor(color_func=shades_of_grey, random_state=1), \
interpolation="bilinear")
plt.axis("off")
plt.figure()


## MOST POSITIVE REVIEW CLUSTER : 5, ALL WORDS AND ALL SENTIMENT WORDS
print("\nThe most cluster number with most positive sentiment is:\n")
pos_cluster=df3['sentiment'].idxmax()
print(pos_cluster)

df4=df2[df2['topic']==pos_cluster]

wrd=" ".join(pd.Series(df4['description']).astype(str))

def shades_of_grey(word, font_size, position, orientation, random_state=None, \
**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)

wcm1 = WordCloud(background_color="green", max_words=200, stopwords=stopw, collocations=False, \
random_state=341)
wcm1.generate(wrd)

plt.imshow(wcm1.recolor(color_func=shades_of_grey, random_state=1), \
interpolation="bilinear")
plt.axis("off")
plt.figure()


## MOST NEGATIVE CLUSTER : 1, ALL WORDS AND ALL SENTIMENT WORDS
print("\nThe most cluster number with least positive sentiment is:\n")
neg_cluster=df3['sentiment'].idxmin()
print(df3['sentiment'].idxmin())

df5=df2[df2['topic']==neg_cluster]

wrd=" ".join(pd.Series(df5['description']).astype(str))

def shades_of_grey(word, font_size, position, orientation, random_state=None,\
**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)

wcm1 = WordCloud(background_color="red", max_words=200, stopwords=stopw, collocations=False,  \
random_state=341)
wcm1.generate(wrd)

plt.imshow(wcm1.recolor(color_func=shades_of_grey, random_state=1), \
interpolation="bilinear")
plt.axis("off")
plt.figure()

print("WordCloud........END")
print("***************************************************************")
