# Import Necessary Libraries
import pandas as pd
from PreProcessor import Preprocessing
from collections import Counter

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from gensim.models import word2vec


# Hyperparameters
batch_size = 8
embedding_dim = 256
hidden_dim = 128
num_layers = 2
num_epochs = 10
learning_rate = 0.001


# Read data
df = pd.read_csv("Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)
df["text"] = df["text"].apply(Preprocessing)


# Label encoding
diseases = df["label"].unique()
# Dictionaries converting diseases to index and vice versa
idx2dis = {k:v for k, v in enumerate(diseases)}
dis2idx = {v:k for k, v in idx2dis.items()}

df["label"] = df["label"].apply(lambda x: dis2idx[x])


# Dataset split 
X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# Reset non-continuoues index of divided dataset
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)


# Create a vocabulary list
def build_vocab(texts):
    # 統計詞頻
    word_freq = Counter()
    for text in texts:
        word_freq.update(text)

    # 根據詞頻排序
    sorted_vocab = sorted(word_freq.items(), key = lambda x: x[1], reverse=True)

    # 去除低詞頻
    #vocab = [word for word, freq in sorted_vocab if freq>=min_freq]

    vocab = {word: i+2 for i, (word, _) in enumerate(sorted_vocab)}
    vocab["PAD"] = 0
    vocab["UNK"] = 1

    idx2word = {v:k for k, v in vocab.items()}

    return vocab, idx2word

vocab, idx2word = build_vocab(X_train)


# Create a PyTorch dataset
max_words = X_train.apply(len).max()    #31

class DiseaseDataset(Dataset):
    def __init__(self, symptoms, labels):
        self.symptoms = symptoms
        self.labels = torch.tensor(labels.to_numpy())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.symptoms[idx]
        label = self.labels[idx]

        # Convert the text to a sequence of word indices
        text_indices = [vocab[word] if word in vocab else vocab["UNK"] 
                        for word in text]

        # Padding for same length sequence
        if len(text_indices)<max_words:
            text_indices = text_indices+[0]*(max_words-len(text_indices))

        return torch.tensor(text_indices), label
    

# Instantiate dataset objects
train_dataset = DiseaseDataset(X_train, y_train)
val_dataset = DiseaseDataset(X_val, y_val)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


def check_accuracy(train_loader, val_loader, model):
    model.eval()
    train_correct = train_total = val_correct = val_total = 0

    with torch.no_grad():
        for x, y in train_loader:
            x = x.squeeze(1)

            _, predictions = model(x).max(1)
            train_correct += (predictions == y).sum()
            train_total += predictions.size(0)

        for x, y in val_loader:
            x = x.squeeze(1)

            _, predictions = model(x).max(1)
            val_correct += (predictions == y).sum()
            val_total += predictions.size(0)

    print(f"Train       Accuracy: {float(train_correct)/float(train_total)*100:.2f}%    ({train_correct}/{train_total})")
    print(f"Validation  Accuracy: {float(val_correct)/float(val_total)*100:.2f}%    ({val_correct}/{val_total})")

    model.train()


# Define BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out[:, -1, :])

        return out
    
model = BiLSTM(len(vocab), embedding_dim, hidden_dim, num_layers, len(diseases))
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


# Training
for epoch in range(num_epochs):
    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.squeeze(1)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print("\n--- Epoch {} ---".format(epoch+1))
    check_accuracy(train_loader, val_loader, model)
