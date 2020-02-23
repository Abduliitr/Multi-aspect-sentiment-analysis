import pandas as pd

## to increase the display width and display it on the same line
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

## function to calculate the average-word-length in a sentence
def avg_word(sentence):
    words = str(sentence).split()
    return (sum(len(word) for word in words)/len(words))

## to read the csv file
train = pd.read_csv('Hoteldata.csv', low_memory=False)

# train = pd.read_csv('sampleData.csv')
# print(train)
# print(train.head())

## to calculate the word-Count in the comments
train['word_Count'] = train['Comment'].apply(lambda x: len(str(x).split(" ")))

## to calculate the char-count in the comments
train['char_Count'] = train['Comment'].str.len() #this will also include blank spaces

## to calculate the average word length of the sentence
train['avg_word']   = train['Comment'].apply(lambda x: avg_word(x))

## to print the array of all the pre-processes...
print(train[['Comment','word_Count','char_Count','avg_word']].head())