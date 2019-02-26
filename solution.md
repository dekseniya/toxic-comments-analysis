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

# Some comments had multiple labling:

 23 


print("Number of rows with multiple labling: ", (rowsums > 1).sum())


# We are mainly interested in frequencies of words for toxic comments, so it is needed to filter out rows which were labled as non-toxic:

# ### Cheking for null values

 24 


print(train.isnull().sum())
print(test.isnull().sum())


### Merging threat, severe_toxic and identity_hate into one category:

### Hovewer these 3 categories aren't correlated with each other, but they also don't have hight correlation with any other category.
### So merging is only based on small numbers of labled values in these categories.

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

# Remove stop words

```
train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

test['comment_text'] = test['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))

train.tail(10)
```


### Tokenization is  breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens.

### Tokenize words:

### Tokenize each comment for adding to data frame

train['tokenized'] = train['comment_text'].apply(word_tokenize)

test['tokenized'] = test['comment_text'].apply(word_tokenize)

### Due to the Wikipedia the length of the longest word in major English dicitonary is 45. So we'll delete all words which are longer than 45, beacuse there are many nonsense words. Also remove all words that have 2 letters or less
 ```
train['tokenized'] = train['tokenized'].apply(lambda x: [item for item in x if 2 < len(item)<45])

test['tokenized'] = test['tokenized'].apply(lambda x: [item for item in x if 2 < len(item)<45])
```

### Lemmatization is process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.

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


    








