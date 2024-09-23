import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model2 import estimate_loss, get_fMRI_batch, fMRI_transformer_model
from transformers import GPT2Tokenizer
import numpy as np
from scipy.io import loadmat, savemat
import h5py
import numpy as np
import matplotlib.pyplot as plt




# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
num_brain_regions = 200 #number of brain regions in the fMRI data, also making this the embedding dimension
n_embd = num_brain_regions
n_head = 4
n_layer = 1
dropout = 0.0 #dropout probability
# ------------

torch.manual_seed(1337)


#load the fMRI data
# open data
data = loadmat('/Users/jacobtanner/Documents/schaefer200_structfunc_data_36pSpike.mat')
data = data['sf_hcp_data']


model = fMRI_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, data, eval_iters, block_size, batch_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

    # sample a batch of data
    batch, target_ts_batch = get_fMRI_batch(data,batch_size, block_size, train_test='train')

    # evaluate the loss
    logits, loss = model(batch, target_ts_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
batch, target_ts_batch = get_fMRI_batch(data,batch_size, block_size, train_test='train')
context = batch[0,:,:].squeeze(0)
predicted_fMRI = m.generate(context, max_new_tokens=200)

#plot the predicted fMRI data
plt.plot(predicted_fMRI.detach().numpy())
plt.show()
