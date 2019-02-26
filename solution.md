#  Toxic Comments Classification

### Import packages
```
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from sklearn.decomposition import LatentDirichletAllocation
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import nltk
from collections import Counter
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, pos_tag_sents
sns.set(style="white", context="talk")
pd.set_option('max_colwidth', -1)
```
## Data loading and preprocessing
```
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
```

### If sum of all rows is equal to zero then a comment is clean.
```
rowsums=train.iloc[:,2:].sum(axis=1)
all_zero =(rowsums==0)

print("Total number of comments in train set: ", len(train))
print("Total number of comments in test set: ", len(test))
print("Total number of clean comments: ", all_zero.sum())
print("Total number of toxic comments: ", rowsums.sum())
```

### Distribtuion of comments by the lable groups in train set:

```
plt.pie(x, labels= y) 
plt.axis('equal')
plt.show()
```

### Number of comments by category (including multi-labeled comments):
```

print("Number of toxic comments: ", train['toxic'].sum())
print("Number of identity hate comments: ", train['identity_hate'].sum())
print("Number of insult comments: ", train['insult'].sum())
print("Number of threat comments: ", train['threat'].sum())
print("Number of obscene comments: ", train['obscene'].sum())
print("Number of severe toxic comments: ", train['severe_toxic'].sum())
print(train.shape)

```

### Some comments had multiple labling:

```
print("Number of rows with multiple labling: ", (rowsums > 1).sum())
```


### We are mainly interested in frequencies of words for toxic comments, so it is needed to filter out rows which were labled as non-toxic:

### Cheking for null values

```
print(train.isnull().sum())
print(test.isnull().sum())
```


### Merging threat, severe_toxic and identity_hate into one category:

Hovewer these 3 categories aren't correlated with each other, but they also don't have hight correlation with any other category.
So merging is only based on small numbers of labled values in these categories.

```
train.corr()

train['merged'] = 0
train.loc[(train['threat'] == 1)
          |
          (train['severe_toxic'] ==1)
          |
          (train['identity_hate'] == 1),
          ['merged']] = 1


(train.loc[(train['severe_toxic']==1) & (train['threat'] ==1) & (train['identity_hate'] ==1)]).shape

```

### Text Presprocessing:

### Replace shortcuts (e.g don't, i'm etc) with full words:

```

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)




train['comment_text'] = train['comment_text'].apply(decontracted)
test['comment_text'] = test['comment_text'].apply(decontracted)

```
### Remove commas, full stops, backslashes, and quotation marks etc.:

```
train['comment_text'] = train['comment_text'].str.replace('\s+|[,.\"!?@|_:;，\(\)\'/...=\►╟\─\-\•✆\[\]\&#~`→+*%><\\\{\}\|]', ' ') 

test['comment_text'] = test['comment_text'].str.replace('\s+|[,.\"!?@|_:;，\(\)\'/...=\►╟\─\-\•✆\[\]\&#~`→+*%><\\\{\}\|]', ' ')  

train.head(50)
```


### Remove digits and change all words to lowercase: 


```
train['comment_text'] = train['comment_text'].str.replace(r'[0-9]', '')

test['comment_text'] = test['comment_text'].str.replace(r'[0-9]', '')


 


train['comment_text'] = train['comment_text'].str.lower()

test['comment_text'] = test['comment_text'].str.lower()
```


### Remove letters, which are appear in word more than two times:


```
def remover(char):
    char = re.sub(r'(.)\1{2,}', r'\1', char)
    return char





train['comment_text'] =  train['comment_text'].apply(remover)

test['comment_text'] =  test['comment_text'].apply(remover)
```


### Define and remove stopwords. 'english' is default list of stopwords from nltk library.

```

stops = list(stopwords.words("english"))
stops.append('u')
stops.append('yep')
stops.append('yea')
stops.append('also')
stops.append('within')
```

### Remove stop words

```
train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

test['comment_text'] = test['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

train.tail(10)
```


### Tokenization is  breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens.

### Tokenize words:

### Tokenize each comment for adding to data frame
```
train['tokenized'] = train['comment_text'].apply(word_tokenize)

test['tokenized'] = test['comment_text'].apply(word_tokenize)
```
### Due to the Wikipedia the length of the longest word in major English dicitonary is 45. So we'll delete all words which are longer than 45, beacuse there are many nonsense words. Also remove all words that have 2 letters or less
 ```
train['tokenized'] = train['tokenized'].apply(lambda x: [item for item in x if 2 < len(item)<45])

test['tokenized'] = test['tokenized'].apply(lambda x: [item for item in x if 2 < len(item)<45])
```

Lemmatization is process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.

```
lemmatizer = WordNetLemmatizer()

test['tokenized'] = test['tokenized'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x])

test['tokenized'] = test['tokenized'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x])

train['tokenize_new']=train['tokenized'].apply(lambda x : " ".join(x))

test['tokenize_new']=test['tokenized'].apply(lambda x : " ".join(x))

```
### Words frequencies:

Count vectorizer will create vocabulary and count number of occurancy of each tokenized word:

(Convert a collection of text documents to a matrix of token counts This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix)

minDf parameter define the minimum number od documents (in this case comments) where term appears.
 
vocabSize specify maximum number of terms in the final vocabulary.

```
no_features = 5000


vectorizer = CountVectorizer(max_features = no_features)

vectorizer.fit(train['tokenize_new'])
train_counts = vectorizer.transform(train['tokenize_new'])

test_counts = vectorizer.transform(test['tokenize_new'])


feature_names = vectorizer.get_feature_names()

print(train_counts.shape)
print(type(train_counts))
```
Measure of how important a word may be is its term frequency (tf) - how frequently a word occurs in a document. Inverse document frequency (idf) decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents.

IDF is computed as log(Total number of documents/Number of documents where specific term appearing)
This can be combined with term frequency to calculate a term’s tf-idf, the frequency of a term adjusted for how rarely it is used. It is intended to measure how important a word is to a document in a collection (or corpus) of documents.

```
frq_vector = TfidfVectorizer(stop_words = 'english', max_features = no_features, ngram_range = (1,1), sublinear_tf=True)
```
tokenize and build vocab
```
train_freq = frq_vector.fit_transform(train['tokenize_new'])
```

tokenize and build vocab
```
test_freq = frq_vector.transform(test['tokenize_new'])
```
###  LDA clusters:

LDA can only use raw term counts for LDA because it is a probabilistic graphical model

Creating LDA clusters from lemmitize words:

'batch': Batch variational Bayes method. Use all training data in  each EM update.
Old `components_` will be overwritten in each iteration.

```
lda = LatentDirichletAllocation(n_components=20, max_iter=20, 
                                learning_method='online', learning_offset=50.,random_state=0)
                                
x_lda = lda.fit_transform(train_counts)

x_lda_test = lda.transform(test_counts)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 20

display_topics(lda, feature_names, no_top_words)
lda_matrix_train = sparse.csr_matrix(x_lda)
lda_matrix_test =  sparse.csr_matrix(x_lda_test)
type(x_lda)
lda_matrix_test.shape
```

### Logistic regression:

The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).

Create matrix of features and target columns. In our aproach we're building different model for each category. So predictions will
be toxic -  non toxic, obscene - non toxic, etc.

 ```

target_col = ['toxic', 'obscene', 'insult', 'merged']
y = train[target_col]
````

Features are TFIDF weights and probabilities from LDA clustering:
```
X = hstack([lda_matrix_train, train_freq])

```

Split train data into train and test sets:
```
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)
```


### Create logistic regression model with hyper prameters tuning:

```
prd = np.zeros((x_test.shape[0],y.shape[1]))
cv_score =[]
for i,col in enumerate(target_col):
    lr = LogisticRegression(class_weight = 'balanced')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    logreg_cv = GridSearchCV(lr, param_grid, cv = 5 )
    print('Building {} model for column:{''}'.format(i,col)) 
    logreg_cv.fit(X_train,y_train[col])
    cv_score.append(logreg_cv.score(x_test, y_test[col]))
    prd[:,i] = logreg_cv.predict_proba(x_test)[:,1]
```

### Print the tuned parameters and score
```
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))





for i, col in enumerate(target_col):
    y_pred = logreg_cv.predict(x_test)
    print('\nConfusion matrix:', col, '\n' ,confusion_matrix(y_test[col],y_pred))
    print(classification_report(y_test[col],y_pred))
    print(logreg_cv.score(x_test, y_test[col]))





print('Overall accuracy', np.mean(cv_score))
```

### Plotting ROC AUC curves for each model:

```
for i, col in enumerate(target_col):
    y_pred_pro = logreg_cv.predict_proba(x_test)[:,1]
 ```
 
### Generate ROC curve values: fpr, tpr, thresholds

```
    fpr, tpr, thresholds = roc_curve(y_test[col], y_pred_pro)
    auc_val =auc(fpr, tpr)
 ```
 
### Plot ROC curve
```
    plt.plot([0, 1], [0, 1], color = 'b')
    plt.plot(fpr,tpr,color='r',label= 'AUC = %.2f'%auc_val)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title(col)
    plt.show()
  ```

 








    








