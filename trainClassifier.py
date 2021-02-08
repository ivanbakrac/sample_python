import pandas as pd
import time
import nltk
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit    
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle

def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    
    reviews_clean=[]
    porter=PorterStemmer()
    for review in reviews:
        review=re.sub('[^A-Za-z0-9 ]+','',review).lower()
        # review=porter.stem(review)
        reviews_clean.append(review)
   

    return reviews_clean,labels

def loadTest(fname):
    reviews=[]
    f = open(fname)
    for line in f:
        review=line.strip().split('\t') 
        #reviews.append(review[0].lower()) 
        
        review = line.strip().rstrip().lstrip()
        reviews.append(review)
    
    for review in reviews:
        review=re.sub('[^A-Za-z0-9 ]+','',review).lower()
    
    f.close()
    
    return reviews

start_time = time.time()


#train,labels=loadData('reviews_train.txt')
#rev_test,labels_test=loadData('reviews_test.txt')


#===============================================================================================
# main training file
#===============================================================================================
import re
f=open('reviews0505_train.txt')
reviews_all=[]
labels_all=[]

for line in f:
    list1=line.strip().split('\t') 
    for i in range(len(list1)-1) :
        if list1[i+1]=='1' or list1[i+1]=='0':
            list1[i]=re.sub('[^A-Za-z0-9 ]+','',list1[i]).lower()
            reviews_all.append(list1[i].lower())
            labels_all.append(int(list1[i+1]))

#===============================================================================================
# train=reviews_all[:50000]
# labels=labels_all[:50000]

labels = labels_all[0:2000]+labels_all[-2000:]
train= reviews_all[0:2000]+reviews_all[-2000:]

#train, labels=loadData('reviews_mv.txt') 
#train, labels=loadData('imdb_labelled.txt')   
#train,labels=loadData('yelp.txt')
#train, labels=loadData('reviews_train.txt')  
#test=loadTest('reviews_test.txt')
#train, labels=loadTest('amazon_cells_labelled.txt')
#train, labels=loadData('amazon_cells_labelled.txt') 



# rev_train,labels_train=loadData('reviews_train.txt')
# rev_test,labels_test=loadData('reviews_test.txt')

# reviews_mv = pickle.load(open('reviews_pkl.pkl', 'rb'))
# labels=reviews_mv.loc[:100,'sentiment'].values
# train=reviews_mv.loc[:100,'review'].values




rev_train,rev_test,labels_train,labels_test=train_test_split(train,labels,test_size=0.5, random_state=0, stratify=labels)




skf = StratifiedKFold(n_splits = 2)
skf.get_n_splits(train, labels)

for train_index, test_index in skf.split(train, labels):
    rev_train, rev_test = [train[i] for i in train_index], [train[i] for i in test_index]
    labels_train, labels_test = [labels[i] for i in train_index], [labels[i] for i in test_index]





from scipy import stats

print(stats.itemfreq(labels_train))
print(stats.itemfreq(labels_test))





def tokenizer(text):
    return text.split()

def tokenzier_porter(text):
    porter=PorterStemmer()
    return [porter.stem(words) for words in text.split()]

text1='ivan bakrac'
print(tokenzier_porter(text1))
   
stop=stopwords.words('english') 

tfidf=TfidfVectorizer()
cvec=CountVectorizer()


print("Training Models..")
#===============================================================================================
# LREG
#===============================================================================================
lreg_param_grid=[{'vect__ngram_range':[(1,1),(1,2)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'clf__penalty':['l1','l2'],
              'clf__C':[0.001,0.01,1.0,10.0,100]},
             {'vect__ngram_range':[(1,1),(1,2)],
              'vect__norm':[None],
              'vect__use_idf':[False],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'clf__penalty':['l1','l2'],
              'clf__C':[0.001,0.01,1.0,10.0,100]}
             ]

lreg_pipe=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0,solver='liblinear'))])
lreg_clfgs=GridSearchCV(lreg_pipe,lreg_param_grid,scoring='accuracy',cv=5,n_jobs=1,verbose=2)
lreg_clfgs_fit=lreg_clfgs.fit(rev_train,labels_train)
lreg_clfgs_best=lreg_clfgs_fit.best_estimator_
lreg_clfgs_best.fit(rev_train,labels_train)

predictedLREG_BEST=lreg_clfgs_best.predict(rev_test)
#===============================================================================================
# NB
#===============================================================================================
nb_param_grid=[{'vect__ngram_range':[(1,1),(1,2)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'vect__max_df':(0.8, 1),
              'vect__min_df':(0.01, 0.05),
              'clf__alpha': [1, 1e-1, 1e-2]},
             {'vect__ngram_range':[(1,1),(1,2)],
              'vect__norm':[None],
              'vect__use_idf':[False],
              'vect__stop_words':[stop,None],
              'vect__max_df':(0.8, 1.00),
              'vect__min_df':(0.01, 0.05),
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'clf__alpha': [1, 1e-1, 1e-2]}
             ]

nb_classifier = Pipeline([('vect', tfidf),('clf', MultinomialNB())])
gridsearch_nb = GridSearchCV(nb_classifier,nb_param_grid, cv=5,scoring='accuracy',n_jobs=1)
gridsearch_nb_fit = gridsearch_nb.fit(rev_train,labels_train)
gridsearch_nb_best=gridsearch_nb_fit.best_estimator_
gridsearch_nb_best.fit(rev_train,labels_train)

predictedNB_BEST=gridsearch_nb_best.predict(rev_test)

#===============================================================================================
# DT
#===============================================================================================
dt_param_grid=[{'vect__ngram_range':[(1,1),(1,2)],
              'vect__stop_words':[stop,None],
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'vect__max_df':(0.8, 1),
              'vect__min_df':(0.01, 0.05),
              'clf__criterion':['gini','entropy'],
              'clf__max_depth':[3,4,5,6,7,8,9,10,11,12]},
             {'vect__ngram_range':[(1,1),(1,2)],
              'vect__norm':[None],
              'vect__use_idf':[False],
              'vect__stop_words':[stop,None],
              'vect__max_df':(0.8, 1.00),
              'vect__min_df':(0.01, 0.05),
              'vect__tokenizer':[tokenizer,tokenzier_porter],
              'clf__criterion':['gini','entropy'],
              'clf__max_depth':[3,4,5,6,7,8,9,10,11,12]}
             ]


dt_classifier = Pipeline([('vect', tfidf),('clf', DecisionTreeClassifier(random_state=0))])
gridsearch_dt = GridSearchCV(dt_classifier,dt_param_grid, cv=5,scoring='accuracy',n_jobs=1)
gridsearch_dt_fit = gridsearch_dt.fit(rev_train,labels_train)
gridsearch_dt_best=gridsearch_dt_fit.best_estimator_
gridsearch_dt_best.fit(rev_train,labels_train)

predictedDT_BEST=gridsearch_dt_best.predict(rev_test)



#===============================================================================================
# VT
#===============================================================================================
eclf1 = VotingClassifier(estimators=[('lr', lreg_clfgs_best), ('nb', gridsearch_nb_best),('dt', gridsearch_dt_best)], voting='hard')
eclf1.fit(rev_train, labels_train)
predictedVT=eclf1.predict(rev_test)
#===============================================================================================
# Cross-Validate with Nested CV:
# Step1 : Grid , CV=5 , inner loop, tune parameters
# Step2: Outer Loop, CV =5 , cross validate on train data across models
#===============================================================================================

# all_clf=[eclf1,gridsearch_dt,gridsearch_nb,lreg_clfgs]
# clf_labels=['vt','dt','nb','lreg']

# for clf,labels in zip(all_clf,clf_labels):
#     scores=cross_val_score(estimator=clf,X=rev_train,y=labels_train,cv=5,scoring='accuracy')
#     print(scores.mean(),labels)

#class_weight='balanced' = another way to balance classes?

#===============================================================================================
print("Training Complete.")


# with open ('labels_test.txt', 'w') as fp: 
#     for label in predictedVT:   
#         fp.write(str(label)+'\n')
# fp.close()


#`estimator.get_params().keys()
print("Reporting Classification Reports...")
vt_acc = accuracy_score(labels_test, predictedVT)
print(classification_report(labels_test, predictedVT))
print(vt_acc)

lreg_acc = accuracy_score(labels_test, predictedLREG_BEST)
print(classification_report(labels_test, predictedLREG_BEST))
print(lreg_clfgs.best_params_)
print(lreg_acc)
print(lreg_clfgs_best.score(rev_test,labels_test))

nb_acc = accuracy_score(labels_test, predictedNB_BEST)
print(classification_report(labels_test, predictedNB_BEST))
print(gridsearch_nb.best_params_)
print(nb_acc)
print(gridsearch_nb_best.score(rev_test, labels_test))

dt_acc = accuracy_score(labels_test, predictedDT_BEST)
print(classification_report(labels_test, predictedDT_BEST))
print(gridsearch_dt.best_params_)
print(dt_acc)
print(gridsearch_dt_best.score(rev_test, labels_test))
print("Report Complete.\n")
accs = [vt_acc, lreg_acc, nb_acc, dt_acc]
names = ["VT", "LREG", "NB", "DT"]

best_i = 0
for i in np.arange(0, len(accs), 1):
	if (accs[i] > accs[best_i]):
		best_i = i


print("Please input the name of the testing file. (Ex. reviews_test.txt)") 
print("This file should be in the same folder as this script.")
fname = input('-->')

rev_test = loadTest(fname) 

predictions = []
if names[best_i] == "LREG":
	lreg_clfgs_best.fit(train, labels)
	predictions = lreg_clfgs_best.predict(rev_test)
if names[best_i] == "DT":
	gridsearch_dt_best.fit(train, labels)
	predictions = gridsearch_dt_best.predict(rev_test)
if names[best_i] == "NB":
	gridsearch_nb_best.fit(train, labels)
	predictions = gridsearch_nb_best.predict(rev_test)
if names[best_i] == "VT":
	eclf1.fit(train, labels)
	predictions = eclf1.predict(rev_test)

outF = open("predictions_output.txt", "w")
for i in predictions:
  # write line to output file
  outF.write(str(i))
  outF.write("\n")
outF.close()
print("Predictions Outputted to predictions_output.txt")

print(stats.itemfreq(labels_train))
print(stats.itemfreq(labels_test))


print("--- %s seconds ---" % (time.time()/60 - start_time/60))




