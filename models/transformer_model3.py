

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import mido
import matplotlib.pyplot as plt
import numpy as np

# The original version of this code comes from a transformer building tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe

import torch
import torch.nn as nn

class CombinedMSEAndL1Loss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedMSEAndL1Loss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        return self.alpha * mse + (1 - self.alpha) * l1



def tensor_to_midi(tensor, output_path, time_resolution=96*2, tempo=500000, loss_type='MSE'):
    """
    Convert a PyTorch tensor representing notes over time back into a MIDI file.
    
    Parameters:
    - tensor: PyTorch tensor representing the MIDI file.
    - output_path: Path to save the output MIDI file.
    - time_resolution: Number of ticks per time step.
    - tempo: Tempo in microseconds per quarter note.
    """
    num_notes, num_time_steps = tensor.shape
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set the tempo
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # Initialize the current time
    last_time_step = 0
    
    # Iterate through the tensor to create MIDI messages
    for time_step in range(num_time_steps):
        for note in range(num_notes):
            velocity = int(tensor[note, time_step].item())
            if velocity > 0 and (time_step == 0 or tensor[note, time_step - 1].item() == 0):
                # Note on event
                delta_time = (time_step - last_time_step) * time_resolution
                if loss_type == 'MSE':
                    track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_time))
                elif loss_type == 'BCE':
                    track.append(mido.Message('note_on', note=note, velocity=100, time=delta_time))
                last_time_step = time_step
            elif velocity == 0 and time_step > 0 and tensor[note, time_step - 1].item() > 0:
                # Note off event
                delta_time = (time_step - last_time_step) * time_resolution
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta_time))
                last_time_step = time_step
    
    # Save the MIDI file
    mid.save(output_path)



def midi_to_tensor(midi_file, loss_type, time_resolution=96*2):
    """
    time resolution of 96 is roughly 100ms or 10 time points per second
    Convert a MIDI file to a PyTorch tensor representing notes over time.
    
    Parameters:
    - midi_file: Path to the MIDI file.
    - time_resolution: Number of ticks per time step.
    
    Returns:
    - tensor: PyTorch tensor representing the MIDI file.
    """
    mid = mido.MidiFile(midi_file)
    
    # Get the resolution (ticks per quarter note)
    resolution = mid.ticks_per_beat
    
    # Get the tempo (microseconds per quarter note)
    tempo = 500000  # Default tempo (120 BPM)
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    
    # Determine the number of time steps
    total_ticks = sum([msg.time for track in mid.tracks for msg in track if not msg.is_meta])
    num_time_steps = total_ticks // time_resolution + 1
    
    # Initialize the matrix with dimensions [num_notes, num_time_steps]
    num_notes = 128  # MIDI notes range from 0 to 127
    matrix = np.zeros((num_notes, num_time_steps), dtype=np.float32)
    
    # Populate the matrix
    note_on_times = {}
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time  # Accumulate delta times to get the actual time point
            time_step = current_time // time_resolution
            if not msg.is_meta:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_times[msg.note] = time_step
                    if loss_type == 'MSE':
                        matrix[msg.note, time_step] = msg.velocity
                    elif loss_type == 'BCE':
                        matrix[msg.note, time_step] = 1
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_on_times:
                        start_time_step = note_on_times.pop(msg.note)
                        matrix[msg.note, start_time_step:time_step] = matrix[msg.note, start_time_step]
    
    # Convert to tensor
    tensor = torch.tensor(matrix, dtype=torch.float)
    return tensor

def load_maestro_dataset(dataset_path, time_resolution= 96*2, dataset_size=500, loss_type='MSE'):   
    """
    Load the Maestro dataset and convert MIDI files to PyTorch tensors.
    
    Parameters:
    - dataset_path: Path to the Maestro dataset.
    - time_resolution: Number of ticks per time step.
    
    Returns:
    - tensors: List of PyTorch tensors representing the MIDI files.
    """
    ii = 0
    tensors = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                if ii > dataset_size:
                    break
                midi_file = os.path.join(root, file)
                tensor = midi_to_tensor(midi_file,loss_type,time_resolution=time_resolution)
                tensors.append(tensor)
                ii += 1
                
    
    return tensors






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
  
#define a get_batch function that will sample from the midi tensor using the batch_size and block_size
def get_batch(data, batch_size, block_size, dataset_size, train_test='train'):
        
    #loop through number of batches and get random subjects ts that is block_size long
    for i in range(batch_size):
        if train_test == 'train': # first half of the dataset is for training and second half is for testing
            #get random song
            song = np.random.randint(0,dataset_size*0.75)
            ts = data[song]
        elif train_test == 'test':
            #get random song
            song = np.random.randint(dataset_size*0.75,dataset_size)
            ts = data[song]
        #get random block
        block = np.random.randint(0,ts.shape[1]-(block_size+1))
        block_ts = ts[:,block:block+block_size]
        target_ts = ts[:,block+1:block+block_size+1]
        #append to batch
        if i == 0:
            batch = np.expand_dims(block_ts.T, axis=0)
            target_ts_batch = np.expand_dims(target_ts.T, axis=0)
        else:
            batch = np.concatenate((batch,np.expand_dims(block_ts.T, axis=0)),axis=0)
            target_ts_batch = np.concatenate((target_ts_batch,np.expand_dims(target_ts.T, axis=0)),axis=0)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.float)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.float)
    return batch, target_ts_batch


@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, dataset_size):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data,batch_size, block_size, dataset_size, train_test=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out, logits


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attention_weights = []
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) #also scaling by head_size so that the attention weights don't become basically one-hot encodings
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        self.attention_weights = wei
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
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
        head_size = 24 #n_embd // n_head
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
class MUSIC_transformer_model(nn.Module):

    def __init__(self,n_embd, n_head, dropout, block_size, n_layer, device, loss_type):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.device = device
        self.attention_weights = []
        if loss_type == 'MSE':
            self.loss_function = CombinedMSEAndL1Loss()
        elif loss_type == 'BCE':
            self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_type = loss_type
       

    def forward(self, idx, targets=None):
        B, T, C = idx.shape

        
        MIDI_embedding = idx # MIDI data is already in a good embedding space
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = MIDI_embedding + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        

        if targets is None:
            loss = None
            if self.loss_type == 'MSE':
                #pass through relu activation function to make it sparse but continuous (just like the data)
                x = F.relu(x)
            
        else:

            if self.loss_type == 'MSE':
                #pass through relu activation function to make it sparse but continuous (just like the data)
                x = F.relu(x)
                #perform MSE loss between predicted and actual time series in shape (B,T,C)
                loss = self.loss_function(x, targets)
            elif self.loss_type == 'BCE':
                #perform BCE loss between predicted and actual time series in shape (B,T,C)
                loss = self.loss_function(x, targets)
      

        return x, loss

    def generate(self, idx, max_new_tokens,plot = False, threshold = 0.5):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            #size of input is block_size x num_brain_regions
            idx_cond = idx[-self.block_size:,:].unsqueeze(0)
            # get the predictions
            next_pred, loss = self(idx_cond)
            
            idx_next = next_pred[:,-1,:].squeeze(0).unsqueeze(0)
            if plot:
                plt.figure()
                plt.imshow(next_pred[0].detach().numpy().squeeze())
                plt.title("Predicted MIDI tensor (during generation)")
                plt.ylabel("Time Step")
                plt.xlabel("Note")
                plt.colorbar()
                plt.show()
            
            #if the loss type is BCE, we need to apply a sigmoid function to the output
            if self.loss_type == 'BCE':
                idx_next = torch.sigmoid(idx_next)
                #then threshold to make binary
                idx_next = torch.where(idx_next > threshold, torch.tensor(1.0), torch.tensor(0.0))

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=0) # (block_size (number of time points), number of notes)
        return idx
