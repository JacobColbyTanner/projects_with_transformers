# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math


# data loading
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device, model_select):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data,
                             block_size, batch_size, device)
            if model_select == 'Context_Heads':
                logits, loss = model(X.T, Y)
            else:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class ContextCTRNN(nn.Module):
    def __init__(self, context_size, vocab_size, dt=None, train_alpha=False):
        super().__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.action_embedding_size = 4*vocab_size
        self.tau = 100

        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.beta_power = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta_mult = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # learn the initial action_map
        self.action_map_init = nn.Parameter(torch.randn(
            self.action_embedding_size*self.context_size))

        # turn tokens into embeddings
        # self.token2embedding = nn.Embedding(vocab_size, self.embedding_size)

        # self.embedding2context = nn.Linear(
        # self.embedding_size, self.memory_size, bias=False)
        # self.embedding2actionable = nn.Linear(
        # self.embedding_size, self.memory_size, bias=False)

        self.embedding2context = nn.Embedding(
            vocab_size, self.context_size)
        self.embedding2actionable = nn.Embedding(
            vocab_size, self.context_size)
        self.context2context_map = nn.Linear(
            self.context_size, self.context_size ** 2)
        self.context2action_map_update = nn.Linear(
            self.context_size, self.context_size * self.action_embedding_size, bias=False)
        self.output_layer = nn.Linear(
            self.action_embedding_size, vocab_size)

    def init_action_map(self, batch_size):
        return self.action_map_init.repeat(batch_size, 1)

    def recurrence(self, input, action_map):

        # input_embedding = self.token2embedding(input)

        context_embedding = self.embedding2context(input)
        actionable_embedding = self.embedding2actionable(input)

        action_map = action_map.view(-1,
                                     self.action_embedding_size, self.context_size)

        action_embedding = torch.bmm(
            action_map, actionable_embedding.unsqueeze(-1)).squeeze(-1)

        action_map_update = self.context2action_map_update(
            context_embedding).view(-1, self.action_embedding_size, self.context_size)

        action_map = action_map + self.alpha * action_map_update

        # Change action_map shape back
        action_map = action_map.view(-1, self.context_size *
                                     self.action_embedding_size)

        output = self.output_layer(action_embedding)

        return output, action_map

    def forward(self, input, action_map=None, num_steps=1):
        if action_map is None:
            action_map = self.init_action_map(input.shape[1])
            action_map = action_map.to(input.device)
        else:
            action_map = action_map

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, action_map = self.recurrence(
                    input[i], action_map)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs


class Context_Heads(nn.Module):
    def __init__(self, context_size, vocab_size, num_heads, **kwargs):
        super().__init__()

        self.heads = nn.ModuleList([ContextCTRNN(
            context_size, vocab_size, **kwargs) for _ in range(num_heads)])

    def forward(self, x, targets, num_steps=1):
        # each head votes on the action to be performed
        # get and stack logits from context heads
        raw_logits = torch.cat(
            [h(x, num_steps=num_steps).unsqueeze(0) for h in self.heads], dim=0)
        # softmax each row of raw_logits and then take the sum of all heads
        logits = torch.sum(F.softmax(raw_logits, dim=0), dim=0)
        # calculate loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss


class ContextRNNNet(nn.Module):
    def __init__(self, context_size, vocab_size, **kwargs):
        super().__init__()
        self.rnn = ContextCTRNN(
            context_size, vocab_size, **kwargs)

    def forward(self, x, targets, num_steps=1):

        logits = self.rnn(x)
        # calculate loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss

    def generate(self, x, max_new_tokens=2000):
        outputs = []
        temp = 2
        for i in range(max_new_tokens):
            logits = self.rnn(x)
            logits = logits.div(temp).exp()
            # sample from multinomial distribution
            x = torch.multinomial(F.softmax(logits[-1].squeeze(), dim=0), 1)
            x = x.unsqueeze(0)
            outputs.append(x)

        outputs = torch.cat(outputs, dim=1)
        return outputs
