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
```
