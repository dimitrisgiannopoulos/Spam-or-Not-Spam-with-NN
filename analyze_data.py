from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# nltk.download('punkt')    #if punkt cannot be found uncomment this command

# Data Sample
#                                                   email  label
# 0      mike bostock said received from trackingNUMBE...      0
# 1      no i was just a little confused because i m r...      0
# 2     this is just an semi educated guess if i m wro...      0
# 3     jm URL justin mason writes except for NUMBER t...      0
# 4     i just picked up razor sdk NUMBER NUMBER and N...      0

# [1500 rows x 2 columns]

# Load data
filepath = 'C:\\Data Mining Project\\spam_or_not_spam\\spam_or_not_spam.csv'
df = pd.read_csv(filepath).dropna()  

pd.value_counts(df['label']).plot(kind="bar", rot=0, grid=True).set_title("label")
plt.show()
print(df.describe())

corpus = np.array(df['email'])
labels = np.array(df['label'])

# Split emails into lists of words
tokenized_sentences = []
for sent in corpus:
    tokenize_word = word_tokenize(sent)
    for word in tokenize_word:
        tokenized_sentences.append(word)

# Make words into numbers
unique_words = set(tokenized_sentences)
vocab_length = len(unique_words)
embedded_sentences = [one_hot(sent, vocab_length) for sent in corpus]

# Make uniform lists of words to input into NN
word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpus, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))
padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')

# Create training/testing set 0.75/0.25
train_df, test_df, train_labels, test_labels = train_test_split(padded_sentences, labels, test_size=0.25)

# Create the NN
model = Sequential()
model.add(Embedding(vocab_length, 20, input_length=length_long_sentence))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# Train the NN (10 epochs)
model.fit(train_df, train_labels, epochs=10, verbose=1)

# Evaluate the NN
# loss, accuracy = model.evaluate(test_df, test_labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))

# Predict labels
predictions = model.predict(test_df)
predictions = np.rint(predictions)

# Evaluate metrics
f1 = f1_score(test_labels, predictions, average='binary')
precision = precision_score(test_labels, predictions, average='binary')
recall = recall_score(test_labels, predictions, average='binary')

print('f1 score: ', f1)
print('Precision score: ', precision)
print('Recall score: ', recall,'\n')