import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(42)

train_padded = np.load("../day5/train_padded.npy")
test_padded = np.load("../day5/test_padded.npy")
y_train_labels = pd.read_csv("../day5/y_train_labels.csv")["Category"]
y_test_labels = pd.read_csv("../day5/y_test_labels.csv")["Category"]

with open("../day5/vocab.json") as f:
    vocab = json.load(f)

print("Train padded shape:", train_padded.shape)
print("Test padded shape:", test_padded.shape)
print("Vocabulary size:", len(vocab))

##fit on the full dataset's categories (not just train) so all 6 classes are always represented consistently
full_df = pd.read_csv("../data/news_dataset.csv")
label_encoder = LabelEncoder()
label_encoder.fit(full_df["Category"])

y_train = label_encoder.transform(y_train_labels)
y_test = label_encoder.transform(y_test_labels)

##long (integer) dtype here since these are vocab index lookups for nn.Embedding, not numeric features
X_train_tensor = torch.tensor(train_padded, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(test_padded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

from torch.utils.data import Dataset, DataLoader

class NewsSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NewsSequenceDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)
test_loader = DataLoader(NewsSequenceDataset(X_test_tensor, y_test_tensor), batch_size=16, shuffle=False)

print("Batches per epoch:", len(train_loader))

import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()  ##standard nn.Module setup, same as day 4

        ##turns each word index (a plain number like 327) into a learnable dense vector of embedding_dim numbers
        ##padding_idx=0 tells it to leave the padding token's vector fixed/meaningless since it's not a real word
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        ##processes the sequence of word vectors one at a time, in order, carrying a hidden state forward as memory
        ##batch_first=True just means our data is shaped (batch, sequence, features) instead of (sequence, batch, features)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        ##final classifier layer, same idea as every output layer so far: hidden_dim numbers in, num_classes raw scores out
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  ##(batch, 20 words) becomes (batch, 20 words, embedding_dim vector each)

        ##runs the whole sequence through the lstm. lstm_out = output at every one of the 20 timesteps (unused here)
        ##hidden = the final hidden state after seeing the whole sequence, cell = the lstm's internal memory state
        lstm_out, (hidden, cell) = self.lstm(embedded)

        final_hidden = hidden[-1]  ##grabs the last layer's final hidden state -- the network's summary of the whole headline
        return self.fc(final_hidden)  ##turns that summary into raw class scores

##vocab_size needs +1 since index 0 (padding) isn't in vocab but still needs its own embedding row
vocab_size = len(vocab) + 1
num_classes = len(label_encoder.classes_)
model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=32, hidden_dim=32, num_classes=num_classes)
print(model)

criterion = nn.CrossEntropyLoss()  ##same as day 4 -- softmax + cross entropy combined, expects raw logits + integer labels
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  ##model.parameters() grabs the embedding, lstm, and fc weights all at once

epochs = 30
model.train()  ##same idea as day 4, though this model doesn't use dropout/batchnorm so it matters less here

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  ##clear old gradients before this batch
        outputs = model(batch_X)  ##forward pass through embedding -> lstm -> fc
        loss = criterion(outputs, batch_y)
        loss.backward()  ##autograd computes every gradient, including back through the lstm's gates
        optimizer.step()  ##adam updates every parameter using those gradients
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")

model.eval()  ##same idea as day 4, though this model doesn't have dropout/batchnorm to switch off
correct = 0
total = 0

with torch.no_grad():  ##no need to track gradients, we're only evaluating
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)  ##picks the highest-scoring class per headline
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

test_accuracy = correct / total
print("\nTest accuracy:", test_accuracy)

torch.save(model.state_dict(), "lstm_classifier_weights.pth")  ##same idea as day 4's saved weights
print("Saved model weights to lstm_classifier_weights.pth")
