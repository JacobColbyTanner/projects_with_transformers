import torch

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.token2embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.token2embedding(x)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(out)
        #calculate cross entropy loss between logits and y
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)

        return logits, loss

    #write code to take in a single input and generate a sequence of outputs
    def generate(self, x, max_new_tokens=2000):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        outputs = []
        temp = 2
        for i in range(max_new_tokens):
            x = self.token2embedding(x)
            out, (h0, c0) = self.lstm(x, (h0, c0))
            logits = self.fc(out)
            logits = logits.div(temp).exp()
            #sample from multinomial distribution
            x = torch.multinomial(F.softmax(logits[-1].squeeze(), dim=0), 1)
            x = x.unsqueeze(0)
            outputs.append(x)
             
        outputs = torch.cat(outputs, dim=1)
        return outputs

        
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y, h=None):
        # Embedding the input
        x = self.embedding(x)
        if h is None:
            h = self.init_hidden(x.size(0))
        
        # Passing through RNN
        out, h = self.rnn(x, h)
        
        # Passing through the fully connected layer
        logits = self.fc(out)  # Only take the output of the last time step

        #calculate cross entropy loss between logits and y
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)

        return logits, loss

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def generate(self, x, max_new_tokens=2000, temperature=2.0):
        h = self.init_hidden(x.size(0)).to(x.device)
        outputs = []

        for _ in range(max_new_tokens):
            x = self.embedding(x)
            out, h = self.rnn(x, h)
            logits = self.fc(out)
            
            # Apply temperature scaling
            logits = logits.div(temperature).exp()
            
            # Sample from the distribution
            probs = F.softmax(logits[-1].squeeze(), dim=0)
            x = torch.multinomial(probs, 1)
            x = x.unsqueeze(0)
            outputs.append(x)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

