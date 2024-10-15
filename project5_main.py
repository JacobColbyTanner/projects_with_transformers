import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Context_RNN import Context_Heads, estimate_loss, get_batch, ContextRNNNet
from models.LSTM_model import LSTMModel, VanillaRNN


model_select = 'LSTM' # 'LSTM' or 'Context_Heads', or 'VanillaRNN'
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 10
n_head = 1
#LSTM/RNN hyperparameters
hidden_size = 1000
num_layers = 1
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

if model_select == 'Context_Heads':
    #model = Context_Heads(n_embd, vocab_size, n_head)
    model = ContextRNNNet(n_embd, vocab_size)
elif model_select == 'LSTM':
    model = LSTMModel(vocab_size, n_embd, hidden_size, num_layers=num_layers)
elif model_select == 'VanillaRNN':
    model = VanillaRNN(vocab_size, n_embd, hidden_size, num_layers=num_layers)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Ensure model is in training mode
model.train()

for iter in range(max_iters):

    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device, model_select)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train',train_data, val_data, block_size, batch_size, device)

  
    # evaluate the loss
    if model_select == 'Context_Heads':
        logits, loss = model(xb.T, yb)
        #print("loss: ", loss.item())
    else:
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()


    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#context = xb[0:1, :100]
#print(decode(context[0].tolist()))
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
