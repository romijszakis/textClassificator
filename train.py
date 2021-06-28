import nltk
import os
import itertools
import string
import random
import pickle

from nltk import classify
from nltk import NaiveBayesClassifier
from nltk import word_tokenize
from nltk.corpus import stopwords

# A stop word list with all NLTK stop words + punctuation symbols
stop_words = set(stopwords.words('dutch'))
all_stop_words = stop_words | set(string.punctuation)
ratings = ['1/10','2/10','3/10','4/10','5/10','6/10','7/10','8/10','9/10','10/10',]
all_stop_words.update(ratings)

"""
Directories for all positive and negative reviews

Directories used when creating program:
DBRD/train/pos/ - Dutch book reviews
aclImdb/train/pos/ - IMDB movie reviews
"""
neg_corpus_root = 'DBRD/train/neg/'
pos_corpus_root = 'DBRD/train/pos/'

documents = []
words = []

# Loops through all positive review files in directory
for filename in os.listdir(pos_corpus_root):
    filtered_text = []
    # Opens and reads all of the text in each file
    f = open(os.path.join(pos_corpus_root, filename), encoding="utf8")
    text = f.read()
    # Splits text into words(tokens) and ads each to a list
    tokens = word_tokenize(text)
    # Loops through all words and adds all words that are not in the stop words list to the filtered text list
    for token in tokens:
        if token not in all_stop_words:
            filtered_text.append(token.lower())
    # Two lists are created, one with all the reviews and their sentiment and another with all the words from all reviews
    documents.append((filtered_text, 'pos'))
    words.append(filtered_text)
 
# Loops through all negative review files in directory
for filename in os.listdir(neg_corpus_root):
    filtered_text = []
    # Opens and reads all of the text in each file
    f = open(os.path.join(neg_corpus_root, filename), encoding="utf8")
    text = f.read()
    # Splits text into words(tokens) and ads each to a list
    tokens = word_tokenize(text)
    #Loops through all words and adds all words that are not in the stop words list to the filtered text list
    for token in tokens:
        if token not in all_stop_words:
            filtered_text.append(token.lower())
     # Two lists are created, one with all the reviews and their sentiment and another with all the words from all reviews
    documents.append((filtered_text, 'neg'))
    words.append(filtered_text)
del filtered_text

# Documents are shuffled randomly to avoid bias
random.shuffle(documents)

""" 
Since words in all words list were added from each review,
they are separated by review, this code removes these separations 
and turns this into a big list of all words found in all reviews.
E.g. (("review1word1","review1word2"...),("review2word1"),(...)) -> ("review1word1","review1word2","review2word1"...)
"""
words = list(itertools.chain(*words))

words = nltk.FreqDist(words)

common_words = list(words.keys()) [:5000]

"""
A function that is used to add a True/False value,
based on if the word is found in the top 5000 most commond words list,
value to each word in each review,
which is necessary for the format of training and test data for the classifier
"""
def find_features(document):
    document_words = set(document)
    features = {}
    for word in common_words:
        features[word] = (word in document_words)
        
    return features

"""
Set with all words and whether they are in the Top 5000 and
also if they are from a positive or negative review
List item example: ("word": True, "pos")
"""
train_data = [(find_features(review), sentiment) for (review, sentiment) in documents]
# test_data = feature_sets[len(feature_sets)//2:]

classifier = NaiveBayesClassifier.train(train_data)

# print("Accuracy is:", classify.accuracy(classifier, test_data))
# print(classifier.show_most_informative_features(15))

# Classifier is saved to .pickle file, which then can be used by test.py on test data
save_classifier = open("dutch_book_classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
