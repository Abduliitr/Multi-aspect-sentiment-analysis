'''
DEALING WITH TEXT DATA
'''

#importing required libraries
import pandas as pd
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
import textblob
from textblob import TextBlob
from textblob import Word
# train = pd.read_csv('./Hoteldata.csv', low_memory=False)
train = pd.read_csv('./sampleData.csv', low_memory=False)



# print('\n\nDATA\n\n')
# print(train.head())


train['word_count'] = train['Comment'].apply(lambda x: len(str(x).split(" ")))


#text with word count
# print('\n\nWORD COUNT\n\n')
train[['Comment','word_count']].head()


# number of characters
# print('\n\nNUMBER OF CHARACTERS\n\n')
train['char_count'] = train['Comment'].str.len() ## this also includes spaces
# print(train[['Comment','char_count']].head())


# define function to calculate the avg word length
def avg_word(sentence):
  words = str(sentence).split()
  return (sum(len(word) for word in words)/len(words))

# print('\n\nAVERAGE WORD LENGTH\n\n')

train['avg_word'] = train['Comment'].apply(lambda x: avg_word(x))
# print(train[['Comment','avg_word']].head())


# print('\n\nNUMBER OF STOP WORDS\n\n')

stop = stopwords.words('english')

train['stopwords'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
# print(train[['Comment','stopwords']].head())


# print('\n\nNUMBER OF SPECIAL CHARACTERS\n\n')
train['hastags'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
train[['Comment','hastags']].head()

# print('\n\nNUMBER OF NUMERICS\n\n')
train['numerics'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# print(train[['Comment','numerics']].head())

train['upper'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
train[['Comment','upper']].head()

print(train.head())
# BASIC-PREPROCESSING
print('\n\BASIC-PRE-PROCESSING\n\n')
# LOWERCASE
train['Comment'] = train['Comment'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
train['Comment'].head()

# REMOVAL OF PUNCTUATION
train['Comment'] = train['Comment'].str.replace('[^\w\s]','')
train['Comment'].head()

# REMOVAL OF STOP WORDS
train['Comment'] = train['Comment'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
train['Comment'].head()

# print(train.head())
# Rare words removal

freq = pd.Series(' '.join(train['Comment']).split()).value_counts()[-10:]
freq = list(freq.index)
# print(freq)
train['Comment'] = train['Comment'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
train['Comment'].head()


#Spelling correction
#TODO
train['Comment'][:5].apply(lambda x: str(TextBlob(x).correct()))


#TOKENIZATION
# print('\n\TOKENIZATION\n\n')
TextBlob(train['Comment'][1]).words
# print(train.head())

# # Stemming
# st = PorterStemmer()
# train['Comment'].apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))

#Lemitization

# print('\n\LEMITIZATION\n\n')
train['Comment'] = train['Comment'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
train['Comment'].head()

# print(train.head())




# BASIC FEATURE
train['word_count'] = train['Comment'].apply(lambda x: len(str(x).split(" ")))


#text with word count
# print('\n\nWORD COUNT\n\n')
train[['Comment','word_count']].head()


# number of characters
# print('\n\nNUMBER OF CHARACTERS\n\n')
train['char_count'] = train['Comment'].str.len() ## this also includes spaces
# print(train[['Comment','char_count']].head())


# define function to calculate the avg word length
def avg_word(sentence):
  words = str(sentence).split()
  return (sum(len(word) for word in words)/len(words))

# print('\n\nAVERAGE WORD LENGTH\n\n')

train['avg_word'] = train['Comment'].apply(lambda x: avg_word(x))
# print(train[['Comment','avg_word']].head())


# print('\n\nNUMBER OF STOP WORDS\n\n')

stop = stopwords.words('english')

train['stopwords'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
# print(train[['Comment','stopwords']].head())


# print('\n\nNUMBER OF SPECIAL CHARACTERS\n\n')
train['hastags'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.startswith('#')]))
# print(train[['Comment','hastags']].head())

# print('\n\nNUMBER OF NUMERICS\n\n')
train['numerics'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))
# print(train[['Comment','numerics']].head())

train['upper'] = train['Comment'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
train[['Comment','upper']].head()

print(train.head())
train.to_csv('output.csv')