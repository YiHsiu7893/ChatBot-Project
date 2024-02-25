import torch.nn as nn
from Attention import attention_block

"""### Original BiLSTM model (without Attention)
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
"""

# Define BiLSTM model (with Attention)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = attention_block(hidden_dim*2)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        att_out = self.attention(output)
        out = self.fc(att_out)

        return out