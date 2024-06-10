import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from Feature_Ext import feat_extr
#from Linguistic_Ext import llm_call
from Attention import attention_block

### BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)

        return out
    

# Process Module: (BiLSTM + Feature Extraction + Attention + LLM)
class Process_Module(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, max_words):
        super(Process_Module, self).__init__()

        learning_rate = 0.001
        self.max_words = max_words

        self.b_model = BiLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.b_model.parameters(), lr=learning_rate)
        self.attention = attention_block(201)
        self.classifier = nn.Linear(201, num_classes, dtype=torch.double)


    def run(self, inputs, texts, labels, mode):
        batch_size = inputs.shape[0]
        inputs = inputs.squeeze(1)

        # forward
        # BiLSTM output
        b_out = self.b_model(inputs)
        padding = torch.zeros(batch_size, self.max_words, 1)
        b_out = torch.cat((b_out, padding), dim=2)

        # Feature Extraction output
        extract_vecs = []
        for text in texts:
            symps = feat_extr(text)
            extract_vec = torch.from_numpy(symps)
            extract_vecs.append(extract_vec)
        extract_vecs = torch.stack(extract_vecs)
        
        # Concatenate the two outputs
        concate_tensor = torch.cat((b_out, extract_vecs), dim=1) #shape = (8, 38, 201)
        # Attention output
        att_out = self.attention(concate_tensor)  #shape = (8, 201)
        input = att_out
        """
        # LLM output
        feature_scores = []
        for text in texts:
            score = llm_call(text)
            print(score)
            feature_scores.append(score)
        feature_scores = np.array(feature_scores)
        feature_scores = torch.from_numpy(feature_scores)

        # Concatenate information from two paths, and provide it to the classifier
        input = torch.cat((att_out, feature_scores), dim=1)
        """
        outputs = self.classifier(input)

        if mode == 'train':
            loss = self.criterion(outputs, labels)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            self.optimizer.step()

        return torch.sigmoid(outputs)
