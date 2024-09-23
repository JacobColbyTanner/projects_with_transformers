
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Original version of this code comes from a transformer building tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe


class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  


def get_fMRI_batch(data,batch_size, block_size, train_test='train'):
    
    #loop through number of batches and get random subjects ts that is block_size long
    for i in range(batch_size):
        #get random subject
        subject = np.random.randint(0,95)
        subject_ts_across_scans = data[0,subject]
        #get random scan, first two scans are train, last 2 are test
        if train_test == 'train':
            scan = np.random.randint(0,2)
        elif train_test == 'test':
            scan = np.random.randint(2,4)
        ts = subject_ts_across_scans['func']['scan'][0][0]['ts'][0][scan]
        #get random block
        block = np.random.randint(0,ts.shape[0]-(block_size+1))
        block_ts = ts[block:block+block_size,:]
        target_ts = ts[block+1:block+block_size+1,:]
        #append to batch
        if i == 0:
            batch = np.expand_dims(block_ts, axis=0)
            target_ts_batch = np.expand_dims(target_ts, axis=0)
        else:
            batch = np.concatenate((batch,np.expand_dims(block_ts, axis=0)),axis=0)
            target_ts_batch = np.concatenate((target_ts_batch,np.expand_dims(target_ts, axis=0)),axis=0)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.float)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.float)
    return batch, target_ts_batch



@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_fMRI_batch(data,batch_size, block_size,train_test=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

# 
# fMRI_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device)
class fMRI_transformer_model(nn.Module):

    def __init__(self,n_embd, n_head, dropout, block_size, n_layer, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.device = device

    def forward(self, idx, targets=None):
        B, T, C = idx.shape

        
        fmri_embedding = idx # fmri data is already in a good embedding space
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = fmri_embedding + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        

        if targets is None:
            loss = None
        else:
            #perform MSE loss between predicted and actual time series in shape (B,T,C)
            loss = F.mse_loss(x, targets)
      

        return x, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            #size of input is block_size x num_brain_regions
            idx_cond = idx[-self.block_size:,:].unsqueeze(0)
            # get the predictions
            next_pred, loss = self(idx_cond)
            
            idx_next = next_pred[:,-1,:].squeeze(0).unsqueeze(0)
            #print("idx shape", idx.shape)
            #print("idx_next shape", idx_next.shape)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=0) # (block_size, num_brain_regions)
        return idx
