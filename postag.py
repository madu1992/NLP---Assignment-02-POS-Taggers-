import numpy as np # linear algebra
import pandas as pd # data processing

from subprocess import check_output

import nltk
from nltk.corpus import brown
from nltk import word_tokenize, pos_tag
nltk.download('brown')
nltk.download('state_union')
nltk.download('punkt')
from nltk.corpus import brown
from nltk.corpus import state_union

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

text = state_union.raw(r"C:\Users\user-pc\Documents\Python\new-testing.txt")
tokens = nltk.word_tokenize(text)

#print("My text : ", text)
#print("My tokens : ", tokens)

from nltk import word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')

"""
Search the max tag in Brown Corpus
"""
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
print("Most common tag is : ", nltk.FreqDist(tags).max())

"""
Now we can create a tagger that tags everything as NN
"""
# Default Tagging
default_tagger = nltk.DefaultTagger('NN')
#print("\nCheck results : ", default_tagger.tag(tokens))

# Performances : 
print("\nPerformance with default tagger : ", default_tagger.evaluate(brown_tagged_sents))

# Pos-Tagging
pos_tagger = nltk.pos_tag(tokens)
print("\nWith POS_TAG : ", pos_tagger)

"""
UniGram-Tagging
"""
from nltk.corpus import brown

# Training
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

# Tag our text
unigram_tagger.tag(tokens)


#Train your own Unigram 

# Create a train and test set
size = int(len(brown_tagged_sents) * 0.8)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Training : 
unigram_tagger = nltk.UnigramTagger(train_sents)

# Evaluate
print ("\nEvaluation 1gram on train set ", unigram_tagger.evaluate(train_sents))
print ("Evaluation 1gram on test set ", unigram_tagger.evaluate(test_sents))

accuracy = unigram_tagger.evaluate(test_sents)
print("Accuracy of 1gram tagger:", accuracy)

#BiGram-Tagging

# Training the bigram tagger on a train set
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)

# Tag our text
bigram_tagger.tag(tokens)

#Train your own Bigram 

# Create a train and test set
size = int(len(brown_tagged_sents) * 0.8)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Training the bigram tagger on a train set
bigram_tagger = nltk.BigramTagger(train_sents)

# Evaluate
print ("\nEvaluation 2gram on train set ", bigram_tagger.evaluate(train_sents))
print ("Evaluation 2gram on test set ", bigram_tagger.evaluate(test_sents))

accuracy = bigram_tagger.evaluate(test_sents)
print("Accuracy of 2gram tagger:", accuracy)

#TriGram-Tagging

# Training the bigram tagger on a train set
Trigram_tagger = nltk.TrigramTagger(brown_tagged_sents)

# Tag our text
Trigram_tagger.tag(tokens)

#Train your own Trigram 

# Create a train and test set
size = int(len(brown_tagged_sents) * 0.8)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Training the bigram tagger on a train set
Trigram_tagger = nltk.TrigramTagger(train_sents)

# Evaluate
print ("\nEvaluation 3gram on train set ", Trigram_tagger.evaluate(train_sents))
print ("Evaluation 3gram on test set ", Trigram_tagger.evaluate(test_sents))

accuracy = Trigram_tagger.evaluate(test_sents)
print("Accuracy of 3gram tagger:", accuracy)

#Mix Default, Unigram and Bigram

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)

print ("\nEvaluation mix default/1G/2G on train set ", t2.evaluate(train_sents))
print ("Evaluation mix default/1G/2G on test set ", t2.evaluate(test_sents))

#Combine Default, Unigram and Bigram

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
t3 = nltk.TrigramTagger(train_sents, backoff=t2)
print ("\nEvaluation mix default/1G/2G/3G on train set ", t3.evaluate(train_sents))
print ("Evaluation mix default/1G/2G/3G on test set ", t3.evaluate(test_sents))

#Perceptron tagger

# Create a train and test set
size = int(len(brown_tagged_sents) * 0.8)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

# Train the model 
from nltk.tag.perceptron import PerceptronTagger
pct_tag = PerceptronTagger(load=False)
pct_tag.train(train_sents)

# Check the performance 
print ("\nEvaluation Own PerceptronTagger on train set ", pct_tag.evaluate(train_sents))
print ("Evaluation Own PerceptronTagger on test set ", pct_tag.evaluate(test_sents))

accuracy = pct_tag.evaluate(test_sents)
print("Accuracy of PerceptronTagger tagger:", accuracy)