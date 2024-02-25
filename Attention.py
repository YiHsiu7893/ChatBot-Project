"""
reference: https://blog.csdn.net/qsmx666/article/details/107118550
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class attention_block(nn.Module):
    def __init__(self, hidden_dim):
        super(attention_block, self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        

    def forward(self, inputs):
        # inputs = (batch_size, max_words, hidden_dim) = (8, 31, 256)

        u = torch.tanh(torch.matmul(inputs, self.w_omega))
        # u = (batch_size, max_words, hidden_dim)

        similarity = torch.matmul(u, self.u_omega)
        # similarity = (batch_size, max_words, 1)

        att = F.softmax(similarity, dim=1)
        # att = (batch_size, max_words, 1)

        att_scores = inputs*att
        # att_scores = (batch_size, max_words, hidden_dim)

        output = torch.sum(att_scores, dim=1)

        return output
