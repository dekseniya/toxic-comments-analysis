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






