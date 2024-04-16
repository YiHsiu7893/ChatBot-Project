import torch.nn as nn
import torch

from sklearn.svm import SVC
from torch.optim import Adam
from Feature_Ext import Extractors
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


class SVM(nn.Module):
    def __init__(self, svc_c):
        super(SVM, self).__init__()
        self.model = SVC(C=svc_c)

    def fit(self, x, y):
        return self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)
    

# Module: (BiLSTM + Feature Extraction + Attention)
class Path2_Module(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, max_words):
        super(Path2_Module, self).__init__()

        learning_rate = 0.001
        self.max_words = max_words

        self.b_model = BiLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.b_model.parameters(), lr=learning_rate)
        self.fe_module = Extractors()
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
                extract_vec = torch.from_numpy(self.fe_module.feat_extr(text))
                extract_vecs.append(extract_vec)
            extract_vecs = torch.stack(extract_vecs)
            
            # Concatenate the two outputs
            concate_tensor = torch.cat((b_out, extract_vecs), dim=1) #shape = (8, 38, 201)

            # Attention and Classifier output
            att_out = self.attention(concate_tensor)  # float or double?
            outputs = self.classifier(att_out)

            if mode == 'train':
                loss = self.criterion(outputs, labels)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

            return outputs
