import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model4 import decode_velocity, get_batch, estimate_loss, Music_transformer_Model, load_maestro_dataset, get_decodings
from models.LSTM_model import LSTMModel
import numpy as np
import time

#option to train starting with a pretrained model
train_from_pretrained = False

# hyperparameters
batch_size = 25 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 200
n_layer = 1
hidden_size = 500
dataset_size = 10

# ------------
output_midi_file_path = '/Users/jacobtanner/Downloads/LSTM_music.midi'


torch.manual_seed(1337)
# Path to the Maestro dataset
dataset_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/'
# Load the dataset and translate each N = dataset_size song into a list of tokens
all_encodings = load_maestro_dataset(dataset_path, dataset_size=dataset_size)

# Flatten the list of lists
flat_encodings = [item for sublist in all_encodings for item in sublist]

# Get the vocab size
vocab_size = np.max(np.array(flat_encodings)) + 1

model = LSTMModel(vocab_size, n_embd, hidden_size, num_layers=n_layer)
model = model.to(device)

if train_from_pretrained:
    model.load_state_dict(torch.load('saved_models/'))

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, all_encodings, eval_iters, block_size, batch_size, dataset_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
        

    # sample a batch of data
    xb, yb = get_batch(all_encodings, batch_size, block_size, dataset_size, train_test='train')

    
    start_time = time.time()
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    total_time = time.time() - start_time
    #print(f"step {iter}: loss {loss:.4f}, time {total_time:.2f} sec")

#save the model
torch.save(model.state_dict(), 'saved_models/LSTM_model_piano.pth')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
encodings = model.generate(context, max_new_tokens=2000)[0].tolist()
get_decodings(encodings, output_midi_file_path)

