#!/usr/bin/env python
# coding: utf-8

# # EXP 1 TOKENIZATION
# 

# In[8]:


#Import required libraries and methods
from typing import List 
from nltk.tokenize import sent_tokenize, word_tokenize,   WordPunctTokenizer, TweetTokenizer, TreebankWordTokenizer,RegexpTokenizer
import re
#Own Function to convert sentence into words
def convert_to_words(s: str) -> List[str]:
    '''Returns the Tokenize list of string that split when " " is encounter
    Params:
        s: input string: str
    Returns:
        List[int]: tokenize list of string 
    '''
    ans = []
    s += ' '
    temp = ''
    for i in s:
        if i == ' ':
            ans.append(temp)
            temp = ''
        else:
            temp += i
    return ans
# #Regex function to tokenize
# def regex(s: str) -> List[str]:
#     word_regex_improved = r"(\w[\w']*\w|\w)"
#     word_matcher = re.compile(word_regex_improved)
#     return word_matcher.findall(s)



text = "Hey, I am amulya and it's a rainy day. How are you? #ok"
print(f'Using Own Function: {convert_to_words(text)}')
print(f'Using NLTK Sentence Tokenization: {sent_tokenize(text)}')
print(f'Using NLTK Word Tokenization: {word_tokenize(text)}')
print(f'Using NLTK WordPunctTokenizer: {WordPunctTokenizer().tokenize(text)}')
tokenizer = RegexpTokenizer("[\w']+")
print(f'Using Regex: {tokenizer.tokenize(text)}')
print(f'Using NLTK Tweet Tokenization: {TweetTokenizer().tokenize(text)}')
print(f'Using NLTK TreeBank Word Tokenization: {TreebankWordTokenizer().tokenize(text)}')


# # EXPERIMENT 2 STEMMING AND LEMMITIZATION

# #PORTER STEMMER

# In[10]:


#Import required libraries and methods 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

stemmer = PorterStemmer() #creating a object of the Porter Class 

words = ["eating", "eats", "eaten", "writing", "writes", "programming", "programs", "history", "finally", "finalized"]
# text = input("Enter an sentence: ")
# tokenized_text = word_tokenize(text)
print("Stemming using Porter Stemming:-")
# for word in tokenized_text:
#     print(f'{word}----->{stemmer.stem(word)}')

for word in words:
    print(f'{word}----->{stemmer.stem(word)}')


# #LANCASTER STEMMER

# In[12]:


from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
# text = input("Enter an sentence: ")
# tokenized_text = word_tokenize(text)
print("Stemming using Lancaster Stemming:-")
# for word in tokenized_text:
#     print(f'{word}----->{lancaster.stem(word)}')

for word in words:
    print(f'{word}----->{lancaster.stem(word)}')


# #RegexpStemmer class
# NLTK has RegexpStemmer class with the help of which we can easily implement Regular Expression Stemmer algorithms. It basically takes a single regular expression and removes any prefix or suffix that matches the expression.

# In[13]:


from nltk.stem import RegexpStemmer
reg_stemmer = RegexpStemmer('ing|s$|e$|able$', min=4) #minimum no of word for applying regex
# text = input("Enter an sentence: ")
# tokenized_text = word_tokenize(text)
print("Stemming using Regexp Stemming:-")
# for word in tokenized_text:
#     print(f'{word}----->{reg_stemmer.stem(word)}')

for word in words:
    print(f'{word}----->{reg_stemmer.stem(word)}')


# #Snowball Stemmer

# In[14]:


from nltk.stem import SnowballStemmer
snowballstemmer = SnowballStemmer('english', ignore_stopwords=False)
# text = input("Enter an sentence: ")
# tokenized_text = word_tokenize(text)
print("Stemming using Snowball Stemming:-")
# for word in tokenized_text:
#     print(f'{word}----->{snowballstemmer.stem(word)}')

for word in words:
    print(f'{word}----->{snowballstemmer.stem(word)}')

Wordnet Lemmatizer
Lemmatization technique is like stemming. The output we will get after lemmatization is called 'lemma', which is a root word rather than root stem. the output of stemming. After lemmatization we will be getting valid word that means the same thing
# In[19]:


import nltk
nltk.download('wordnet')


# In[21]:



# import these modules
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
 
print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))
 
# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))


# In[22]:


import nltk 
from nltk.stem import WordNetLemmatizer
lammetizer = WordNetLemmatizer()
# text = input("Enter an sentence: ")
# tokenized_text = word_tokenize(text)
print("Lemmatizing using WordNetLemmatizer Lammetizer:-")
# for word in tokenized_text:
#    print(f'{word}----->{lammetizer.lemmatize(word)}')

for word in words:
    print(f'{word}----->{lammetizer.lemmatize(word)}')


# In[23]:


for word in words:
    print(word + '----->' + lammetizer.lemmatize(word, pos='v'))


# In[24]:


lammetizer.lemmatize('better', pos='a')


# # EXPERIMENT 3  STOPWORDS

# In[25]:


import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))


# In[26]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sent = "This is a sample sentence, showing off the stop words filtration."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(f"Sample Sentence: {example_sent}")
print(f"Tokenized Sentence: {word_tokens}")
print(f"After removing stop words: {filtered_sentence}")


# CREATING OWN STOPWORDS

# In[29]:


example_sent = "This is a sample sentence, showing off the stop words filtration."
stopwords = {'hers', 'ourselves', 'is', 'she', 'her', 'in', 'if', 'a', 'my', 'off'}
word_tokenized = word_tokenize(example_sent)
stop_word_remover = [w for w in word_tokenized if not w in stopwords]
print(f"Sample Sentence: {example_sent}")
print(f"Tokenized Sentence: {word_tokenized}")
print(f"After removing stop words: {stop_word_remover}")


# In[30]:


def remove_custom_stopwords(text, custom_stopwords):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove custom stopwords from the text
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]
    
    # Join the filtered words back into a sentence
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

# Example usage
text = "This is an example sentence to demonstrate stopwords removal."
custom_stopwords = ['is', 'an', 'to']
filtered_text = remove_custom_stopwords(text, custom_stopwords)
print("Original Text:", text)
print("Filtered Text:", filtered_text)


# 
# # EXPERIMENT 4 NGRAMS

# In[41]:


def generate_ngrams(text, n):
    # Convert the text to lowercase and remove any punctuation
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())

    # Tokenize the text into words
    words = text.split()

    # Generate n-grams from the words
    ngrams_list = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams_list.append(ngram)

    return ngrams_list

# Example usage
text = input("Enter the sentence: ")
n = int(input("Enter the value of n: "))

result = generate_ngrams(text, n)
print("Original Text:", text)
print(f"{n}-grams:", result)


# # Experiment 5 n gram smoothing add one method

# P(w_n | w_1, w_2, ..., w_{n-1}) = (Count(w_1, w_2, ..., w_n) + 1) / (Count(w_1, w_2, ..., w_{n-1}) + V)
# 
# 
# Laplace smoothing, also known as add-one smoothing, is a simple technique used in n-gram language modeling to address the problem of zero probabilities for unseen n-grams. It is named after the mathematician Pierre-Simon Laplace.
# 
# In n-gram language models, probabilities are calculated based on the frequency of n-grams observed in the training data. However, there may be cases where certain n-grams do not appear in the training data, resulting in zero probabilities when using them for language modeling tasks.
# 
# Laplace smoothing addresses this issue by adding a small constant value (usually 1) to both the numerator and denominator of the probability calculation for each n-gram. This ensures that even unseen n-grams have a non-zero probability assigned to them.
# 
# The formula for calculating the Laplace-smoothed probability of an n-gram is:
# P(w_n | w_1, w_2, ..., w_{n-1}) = (Count(w_1, w_2, ..., w_n) + 1) / (Count(w_1, w_2, ..., w_{n-1}) + V)
# 

# In[46]:


from collections import defaultdict

def train_ngram_model(corpus, n):
    ngrams_counts = defaultdict(int)
    context_counts = defaultdict(int)

    # Tokenize the corpus into n-grams
    ngrams = [tuple(corpus[i:i+n]) for i in range(len(corpus)-n+1)]

    # Count occurrences of n-grams and their contexts
    for ngram in ngrams:
        context = ngram[:-1]
        ngrams_counts[ngram] += 1
        context_counts[context] += 1

    return ngrams_counts, context_counts

def calculate_smoothed_probability(ngram, ngrams_counts, context_counts, vocabulary_size):
    context = ngram[:-1]
    numerator = ngrams_counts[ngram] + 1
    denominator = context_counts[context] + vocabulary_size

    return numerator / denominator

# Get user input
sentence = input("Enter a sentence: ")
n = int(input("Enter the value of n for n-grams: "))

# Preprocess the sentence
corpus = sentence.lower().split()

# Train the n-gram model
ngrams_counts, context_counts = train_ngram_model(corpus, n)

# Calculate the smoothed probability for each n-gram
vocabulary_size = len(set(corpus))
probabilities = []

for ngram in ngrams_counts.keys():
    probability = calculate_smoothed_probability(ngram, ngrams_counts, context_counts, vocabulary_size)
    probabilities.append((ngram, probability))

# Print the n-grams and their corresponding probabilities
for ngram, probability in probabilities:
    print("N-gram:", ngram)
    print("Smoothed Probability:", probability)
    print("---")


# # Experiment 6 POS USING HMM

# In[50]:


from nltk.tag import hmm

# Prepare the training data
tagged_sentences = [
    [('apple', 'Noun'), ('eats', 'Verb'), ('red', 'Adjective')],
    [('red', 'Adjective'), ('apple', 'Noun'), ('eats', 'Verb')]
]

# Create the training corpus
train_corpus = [[(word, tag) for word, tag in sentence] for sentence in tagged_sentences]

# Create and train the HMM model
trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(train_corpus)

# Perform POS tagging on a new sentence
sentence = ['the', 'red', 'apple', 'eats']
predicted_states = model.best_path(sentence)

# Print the sentence and predicted tags
print("Sentence:", sentence)
print("Tags:", predicted_states)


# # EXPERIMENT 9 CHUNKING TREE

# In[53]:


import nltk
from nltk import Tree

# Define a sample sentence
sentence = "I am Amulya Maurya"

# Tokenize the sentence into words
tokens = nltk.word_tokenize(sentence)

# Perform part-of-speech tagging
pos_tags = nltk.pos_tag(tokens)

# Define a chunk grammar using regular expressions
chunk_grammar = r"""
    NP: {<DT|JJ|NN.*>+}  # Noun phrase
    VP: {<VB.*><NP|PP|CLAUSE>+$}  # Verb phrase
    PP: {<IN><NP>}  # Prepositional phrase
    CLAUSE: {<NP><VP>}  # Clause
"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(chunk_grammar)

# Apply chunking to the part-of-speech tagged sentence
chunked_sentence = chunk_parser.parse(pos_tags)

# Print the chunked sentence
print(chunked_sentence)

# Draw and display the parse tree
chunked_sentence.draw()


# In[60]:


import nltk
from nltk.corpus import movie_reviews
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import random

# Load the movie reviews dataset
#nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents)

# Rest of the code remains the same
# ...

# Extract features from the documents
all_words = FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the dataset into training and testing sets
train_set = featuresets[:1500]
test_set = featuresets[1500:]

# Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Define a sample text
text = "I love this movie."

# Tokenize the sample text
tokens = word_tokenize(text.lower())

# Extract features from the tokens
features = document_features(tokens)

# Perform sentiment analysis using the trained classifier
sentiment = classifier.classify(features)

# Print the sentiment
print("Sentiment:", sentiment)


# In[ ]:




