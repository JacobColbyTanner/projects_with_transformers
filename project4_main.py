import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model4 import decode_velocity, get_batch, estimate_loss, Music_transformer_Model, load_maestro_dataset, get_decodings
import numpy as np
import time

#option to train starting with a pretrained model
train_from_pretrained = False

# hyperparameters
batch_size = 25 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_head = 8
n_embd = 256 #n_head*256 #must be a multiple of n_head, or the head size calculation must be changed in the model
n_layer = 2
dropout = 0.1
dataset_size = 10
use_relational_position_embedding = False
use_relational_time_and_pitch_embeddings = True
use_positional_encoding = False
# ------------
output_midi_file_path = '/Users/jacobtanner/Downloads/generated_music_relative_attention33.midi'


torch.manual_seed(1337)
# Path to the Maestro dataset
dataset_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/'
# Load the dataset and translate each N = dataset_size song into a list of tokens
all_encodings = load_maestro_dataset(dataset_path, dataset_size=dataset_size)

# Flatten the list of lists
flat_encodings = [item for sublist in all_encodings for item in sublist]

# Get the vocab size
vocab_size = np.max(np.array(flat_encodings)) + 1


model = Music_transformer_Model(n_embd, n_head, dropout, vocab_size, block_size, n_layer, 
                                device, use_relational_position=use_relational_position_embedding, 
                                use_relational_time_pitch=use_relational_time_and_pitch_embeddings,use_positional_encoding = use_positional_encoding)
model = model.to(device)

if train_from_pretrained:
    model.load_state_dict(torch.load('saved_models/model4_music_transformer_yes_relative_positional_embedding_ver2.pth'))

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
    print(f"step {iter}: loss {loss:.4f}, time {total_time:.2f} sec")

#save the model
torch.save(model.state_dict(), 'saved_models/model4_music_transformer_yes_relative_positional_embedding_ver2.pth')

# generate from the model
context = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
encodings = model.generate(context, max_new_tokens=2000)[0].tolist()
get_decodings(encodings, output_midi_file_path)

