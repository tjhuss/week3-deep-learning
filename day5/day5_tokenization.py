##scraped more data so from 111->278 data points to work with since these models are more data hungry, will add more going into the week 4 work

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/news_dataset.csv") ##load the expanded 278-row dataset

##same split as every prior day: 80% train, 20% test, same random_state so results stay comparable
X_train_text, X_test_text, y_train_labels, y_test_labels = train_test_split(
    df["Title"], df["Category"], test_size=0.2, random_state=42
)

print("Train shape:", X_train_text.shape)
print("Test shape:", X_test_text.shape)

##tokenizing = breaking a headline into a list of individual words
def tokenize(text):
    return text.lower().split() ##lowercase first so "Stock" and "stock" count as the same word, then split on whitespace

sample_tokens = tokenize(X_train_text.iloc[0]) ##.iloc[0] grabs the first headline in the training set by position
print("Sample headline:", X_train_text.iloc[0])
print("Tokens:", sample_tokens) ##should print as a list of separate words instead of one long string

train_tokens = [tokenize(title) for title in X_train_text] ##all headlines in the training set will undergo tokenizations and become 222 list of tokens or words for each headline
test_tokens = [tokenize(title) for title in X_test_text]##same applies to the test set for the 56 headlines

print("Number of tokenized train headlines:", len(train_tokens))
print("First train example:", train_tokens[0])

##building the vocab for the train set
all_words = set()               ##creates an empty set (refuses any duplicates)
for tokens in train_tokens:
    all_words.update(tokens)    ##each list of lists from the headlines is dumped into all_words and contains each unique word across the training set with the duplicated condensed

vocab = {word: idx + 1 for idx, word in enumerate(sorted(all_words))} ##the sorting function places everything in aplhabetical order since there is no fixed order present, makes it reproducible and consistent for results
##enumerate walks through the sorted list and gives the poistion from (0) and the word itself ex: (67, six-seven)
##idx +1 makes the vocab dictionary, each word is unique key

print("Vocabulary size:", len(vocab))
print("Sample mappings:", list(vocab.items())[:10])

##convert the list of words into a list of numbers for the vocab

def encode(tokens, vocab):
    return[vocab.get(word, 0) for word in tokens] ##looks up each words number, if word is not found then ir returns 0 instead of crashing

train_encoded = [encode(tokens, vocab) for tokens in train_tokens] ##encode every training headline into numbers
test_encoded = [encode(tokens, vocab) for tokens in test_tokens]   ##encode every test headline (may contain words never seen in training, hence the 0 fallback above)

print("First train example (words):", train_tokens[0])
print("First train example (encoded)", train_encoded[0])

##headlines are different lengths (2-24 words), but a network needs every input the same size
max_length = 20

def pad_sequence(sequence, max_length):
    if len(sequence) >= max_length:
        return sequence[:max_length]  ##truncate anything too long
    return sequence + [0] * (max_length - len(sequence))  ##pad anything too short with 0s

train_padded = [pad_sequence(seq, max_length) for seq in train_encoded]
test_padded = [pad_sequence(seq, max_length) for seq in test_encoded]

print("First train example (padded):", train_padded[0])
print("Length check:", len(train_padded[0]))

##saving the actual tokenized dataset to disk -- this is today's real deliverable, not just printed output
import numpy as np
import json

np.save("train_padded.npy", np.array(train_padded))
np.save("test_padded.npy", np.array(test_padded))
y_train_labels.to_csv("y_train_labels.csv", index=False)
y_test_labels.to_csv("y_test_labels.csv", index=False)

with open("vocab.json", "w") as f:
    json.dump(vocab, f)

print("\nSaved train_padded.npy, test_padded.npy, y_train_labels.csv, y_test_labels.csv, vocab.json")