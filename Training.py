# Import Necessary Libraries
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from Models import Process_Module, Process_Module2
from Tokenizers import tokenizer
from Metrics import check_metrics


class DiseaseDataset(Dataset):
    def __init__(self, symptoms, descriptions, labels):
        self.symptoms = symptoms
        self.descriptions = descriptions
        self.labels = torch.tensor(labels.to_numpy())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        text = self.symptoms[idx]
        des = self.descriptions[idx]
        label = self.labels[idx]

        # Convert the text to a sequence of word indices
        text_indices = [vocab[word] if word in vocab else vocab["UNK"] 
                        for word in text]

        # Padding for same length sequence
        if len(text_indices)<max_words:
            text_indices = text_indices+[0]*(max_words-len(text_indices))

        return torch.tensor(text_indices), des, label

# Hyperparameters
batch_size = 8
embedding_dim = 256
hidden_dim = 100
num_layers = 2
num_epochs = 2
fold = 5

# Read data
df = pd.read_csv("Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)


# Label encoding
diseases = df["label"].unique()
idx2dis = {k:v for k, v in enumerate(diseases)}
dis2idx = {v:k for k, v in idx2dis.items()}

df["label"] = df["label"].apply(lambda x: dis2idx[x])

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)
X_total = df["text"]
y_total = df["label"]

# Dataset split 
# X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# Create a vocabulary list
X_total_tk = X_total.apply(tokenizer)

word_freq = Counter()
for text in X_total_tk:
    word_freq.update(text)
vocab = {word: i+2 for i, word in enumerate(word_freq)}
vocab["PAD"] = 0
vocab["UNK"] = 1

max_words = X_total.apply(len).max()    #31


train = [0.0] * 5
val = [0.0] * 5
train2 = [0.0] * 5
val2 = [0.0] * 5
for sp in range(fold):
    # for cross validation
    X_train = X_total.drop(X_total.index[sp*len(X_total)//fold:(sp+1)*len(X_total)//fold])
    X_val = X_total[sp*len(X_total)//fold:(sp+1)*len(X_total)//fold]
    y_train = y_total.drop(y_total.index[sp*len(y_total)//fold:(sp+1)*len(y_total)//fold])
    y_val = y_total[sp*len(y_total)//fold:(sp+1)*len(y_total)//fold]

    # Reset non-continuoues index of divided dataset
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)

    X_train_ori = X_train
    X_train_tk = X_train.apply(tokenizer)
    X_val_ori = X_val
    X_val_tk = X_val.apply(tokenizer)


    # Instantiate dataset objects
    train_dataset = DiseaseDataset(X_train_tk, X_train_ori, y_train)
    val_dataset = DiseaseDataset(X_val_tk, X_val_ori, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    module = Process_Module(len(vocab),  embedding_dim, hidden_dim, num_layers, len(diseases), max_words) 
    module2 = Process_Module2(len(vocab),  embedding_dim, hidden_dim, num_layers, len(diseases), max_words) 

    # Training
    for epoch in range(num_epochs):
        print("\n--- Epoch {} ---".format(epoch+1))

        module.b_model.train()
        module2.b_model.train()
        for inputs, texts, labels in train_loader:
            module.run(inputs, texts, labels, 'train')
            module2.run(inputs, texts, labels, 'train')
        
    #train_apd, val_apd = check_metrics(train_loader, val_loader, module, 1 if epoch+1==num_epochs else 0)
    train_apd, val_apd = check_metrics(train_loader, val_loader, module, 0)
    train_apd2, val_apd2 = check_metrics(train_loader, val_loader, module2, 0)
    for i in range(5):
        train[i] += train_apd[i]
        val[i] += val_apd[i]

        train2[i] += train_apd2[i]
        val2[i] += val_apd2[i]

print(f"Train Average Accuracy: {train[0]/fold:.3f}")
print(f"Train Average Precision: {train[1]/fold:.3f}")
print(f"Train Average Recall: {train[2]/fold:.3f}")
print(f"Train Average F1-score: {train[3]/fold:.3f}")
print(f"Train Average AUC-score: {train[4]/fold:.3f}")
print("==============================")
print(f"Val Average Accuracy: {val[0]/fold:.3f}")
print(f"Val Average Precision: {val[1]/fold:.3f}")
print(f"Val Average Recall: {val[2]/fold:.3f}")
print(f"Val Average F1-score: {val[3]/fold:.3f}")
print(f"Val Average AUC-score: {val[4]/fold:.3f}")

print("\n\n Without Feat_ext \n\n")

print(f"Train Average Accuracy: {train2[0]/fold:.3f}")
print(f"Train Average Precision: {train2[1]/fold:.3f}")
print(f"Train Average Recall: {train2[2]/fold:.3f}")
print(f"Train Average F1-score: {train2[3]/fold:.3f}")
print(f"Train Average AUC-score: {train2[4]/fold:.3f}")
print("==============================")
print(f"Val Average Accuracy: {val2[0]/fold:.3f}")
print(f"Val Average Precision: {val2[1]/fold:.3f}")
print(f"Val Average Recall: {val2[2]/fold:.3f}")
print(f"Val Average F1-score: {val2[3]/fold:.3f}")
print(f"Val Average AUC-score: {val2[4]/fold:.3f}")

torch.save(module.state_dict(), 'model.pth')
torch.save(vocab, 'vocab.pth')
torch.save(idx2dis, 'idx.pth')
