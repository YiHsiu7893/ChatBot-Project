# Import Necessary Libraries
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from Model import BiLSTM
from PreProcessor import normal_preprocess
from Metrics import check_metrics


# Hyperparameters
batch_size = 8
embedding_dim = 256
hidden_dim = 128
num_layers = 2
num_epochs = 5
learning_rate = 0.001


# Read data
df = pd.read_csv("Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)
df["text"] = df["text"].apply(normal_preprocess)


# Label encoding
diseases = df["label"].unique()
idx2dis = {k:v for k, v in enumerate(diseases)}
dis2idx = {v:k for k, v in idx2dis.items()}

df["label"] = df["label"].apply(lambda x: dis2idx[x])


# Dataset split 
X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# Reset non-continuoues index of divided dataset
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)


# Create a vocabulary list
word_freq = Counter()
for text in X_train:
    word_freq.update(text)
vocab = {word: i+2 for i, word in enumerate(word_freq)}
vocab["PAD"] = 0
vocab["UNK"] = 1


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
    check_metrics(train_loader, val_loader, model, 1 if epoch+1==num_epochs else 0)

torch.save(model.state_dict(), 'model.pth')
torch.save(vocab, 'vocab.pth')
torch.save(idx2dis, 'idx.pth')