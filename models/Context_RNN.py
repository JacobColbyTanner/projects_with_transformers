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
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
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
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        self.beta_power = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta_mult = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        #turn tokens into embeddings
        #self.token2embedding = nn.Embedding(vocab_size, self.embedding_size)
        
        #self.embedding2context = nn.Linear(
            #self.embedding_size, self.memory_size, bias=False)
        #self.embedding2actionable = nn.Linear(
            #self.embedding_size, self.memory_size, bias=False)

        

        self.embedding2context = nn.Embedding(
            vocab_size, self.context_size)
        self.embedding2actionable = nn.Embedding(
            vocab_size, self.context_size)
        self.context2context_map = nn.Linear(
            self.context_size, self.context_size ** 2)
        self.context2action_map = nn.Linear(
            self.context_size, self.context_size * self.vocab_size, bias=False)
        
        

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):

        #input_embedding = self.token2embedding(input)

        context_embedding = self.embedding2context(input)
        actionable_embedding = self.embedding2actionable(input)

        #context_map = self.context2context_map(context).view(-1, self.context_size, self.context_size)
        
        transformed_context_embedding =  context_embedding#torch.bmm(
            #context_map, context_embedding.unsqueeze(-1)).squeeze(-1)

        transformed_context_embedding_norm = torch.norm(
            transformed_context_embedding, p=2, dim=1).unsqueeze(1)
        context_norm = torch.norm(context, p=2, dim=1).unsqueeze(1)

        

        beta = self.alpha * (self.beta_mult * (transformed_context_embedding_norm /
                             (transformed_context_embedding_norm + context_norm))) ** self.beta_power

        
        # clamp beta to be between 0 and 1
        beta = torch.clamp(beta, 0, 1)
        

        

        context = (1-beta) * context + beta * transformed_context_embedding
        
        # Compute action map from context
        action_map = self.context2action_map(
            context).view(-1, self.vocab_size, self.context_size)
        
        
        # Ensure actionable_portion has the correct shape for batch matrix multiplication
        actionable_embedding = actionable_embedding.view(-1, self.context_size, 1)
        
        # Apply action map to input
        output = torch.bmm(
            action_map, actionable_embedding).squeeze(-1)


        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1])
            context = context.to(input.device)
        else:
            context = context

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(
                    input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs


        
        
class Context_Heads(nn.Module):
    def __init__(self, context_size, vocab_size, num_heads, **kwargs):
        super().__init__()
        
        self.heads = nn.ModuleList([ContextCTRNN(
            context_size, vocab_size, **kwargs) for _ in range(num_heads)])
   

    def forward(self, x, targets, num_steps=1):
        #each head votes on the action to be performed
        #get and stack logits from context heads
        raw_logits = torch.cat([h(x, num_steps=num_steps).unsqueeze(0) for h in self.heads], dim=0)  
        #softmax each row of raw_logits and then take the sum of all heads
        logits = torch.sum(F.softmax(raw_logits, dim=0),dim=0)
        #calculate loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss



class ContextRNNNet(nn.Module):
    def __init__(self,context_size, vocab_size, **kwargs):
        super().__init__()
        self.rnn = ContextCTRNN(
            context_size, vocab_size, **kwargs)

    def forward(self, x, targets, num_steps=1):

        logits = self.rnn(x)
        #calculate loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss
    
    def generate(self, x, max_new_tokens=2000):
        outputs = []
        temp = 2
        for i in range(max_new_tokens):
            logits = self.rnn(x)
            logits = logits.div(temp).exp()
            #sample from multinomial distribution
            x = torch.multinomial(F.softmax(logits[-1].squeeze(), dim=0), 1)
            x = x.unsqueeze(0)
            outputs.append(x)
             
        outputs = torch.cat(outputs, dim=1)
        return outputs
