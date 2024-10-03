import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_new_attention_mechanism import get_batch, estimate_loss, BigramLanguageModel
from transformers import GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



# TODO
# 1. value matrix with the transform
# 2. normalization on the output of each head (extract the magnitudes of each C dimensional T by T set of vectors, and use to normalize the vectors. Apply softmax to the magnitudes as you would attention weights)



# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 1
n_layer = 4
dropout = 0.1
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()



# Tokenize the text
vocab_size = tokenizer.vocab_size
tokens = tokenizer.encode(text)
encode = lambda s: tokenizer.encode(s) # encoder: take a string, output a list of integers
decode = lambda l: tokenizer.decode(l) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(n_embd, n_head, dropout, vocab_size, block_size, n_layer, device)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train',train_data, val_data, block_size, batch_size, device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
