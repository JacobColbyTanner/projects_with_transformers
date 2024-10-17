
import torch
import torch.nn as nn
from torch.nn import functional as F
import mido
import numpy as np
import os
import matplotlib.pyplot as plt

# The original version of this code comes from a transformer building tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe


#This type of Midi encoding from: 
#Oore, S., Simon, I., Dieleman, S., Eck, D., & Simonyan, K. (2020). This time with feeling: Learning expressive musical performance. Neural Computing and Applications, 32, 955-967.
# Event indices
NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = 128
VELOCITY_OFFSET = 256
TIME_SHIFT_OFFSET = 288

def encode_velocity(velocity):
    # Create 32 bins for the velocities of size 4
    velocity_bins = torch.linspace(0, 127, 32)
    digitized_velocity = torch.bucketize(torch.tensor(velocity), velocity_bins)
    velocity_encoding = digitized_velocity + VELOCITY_OFFSET
    return velocity_encoding.to(torch.int64)

def encode_time_shift(time):
    # Divide time by how many increments of 8ms have passed (this is roughly 8ms given that time is in 0.96 ms here)
    digitized_time = torch.round(torch.tensor(time) / 8)
    #cap the max value of digitized time
    digitized_time = torch.clamp(digitized_time, 0, 200)
    time_encoding = digitized_time + TIME_SHIFT_OFFSET
    return time_encoding.to(torch.int64) 

def encode_note_off(note):
    note = torch.tensor(note) + NOTE_OFF_OFFSET
    return note.to(torch.int64)


def get_encodings(midi_file_path):
    #translates midi file to a list of tokens (integers)
    #example midi_file_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/2017/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.midi'  # Replace with your MIDI file path
    midi_file = mido.MidiFile(midi_file_path)

    all_encodings = []
    for track in midi_file.tracks:
        for msg in track:
            if not msg.is_meta:
                if msg.type == 'note_on' and msg.velocity > 0:
                    all_encodings.append(encode_time_shift(msg.time)) # Time shift event
                    all_encodings.append(torch.tensor(msg.note).to(torch.int64))  # Note on event
                    all_encodings.append(encode_velocity(msg.velocity))  # Velocity event
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    all_encodings.append(encode_time_shift(msg.time)) # Time shift event
                    all_encodings.append(encode_note_off(msg.note)) # Note off event
                    all_encodings.append(encode_velocity(msg.velocity)) # Velocity event
                    
    return all_encodings

def decode_velocity(velocity_encoding):
    # Reverse the encoding process
    digitized_velocity = velocity_encoding - VELOCITY_OFFSET
    velocity_bins = torch.linspace(0, 127, 32)
    velocity = velocity_bins[digitized_velocity]  # 
    return int(velocity.item())

def decode_time_shift(time_encoding):
    # Reverse the encoding process
    digitized_time = time_encoding - TIME_SHIFT_OFFSET
    time = digitized_time * 8  # Multiply by 8 to get the original time in 0.96 ms units
    return time


def get_decodings(encodings, output_midi_file_path):
    # Create a new MIDI file
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    note = 0
    velocity = 0
    time = 0
    A, B, C, D = 0, 0, 0, 0
    for token in encodings:
        if token >= TIME_SHIFT_OFFSET:
            time = decode_time_shift(token)
            A = 1
        elif token >= VELOCITY_OFFSET:
            velocity = decode_velocity(token)
            if velocity == 0:
                velocity = 2
            B = 1
        elif token >= NOTE_OFF_OFFSET:
            note = token - NOTE_OFF_OFFSET
            C = 1
        else:
            note = token
            D = 1
        if A+B+C+D == 3: #check to make sure you have a full message before writing it
            track.append(mido.Message('note_on', note=int(note), velocity=int(velocity), time=int(time)))
            A, B, C, D = 0, 0, 0, 0

    # Save the MIDI file
    midi_file.save(output_midi_file_path)
    

    # Save the MIDI file

def load_maestro_dataset(dataset_path, dataset_size=500):   
    """
    Load the Maestro dataset and encode MIDI data.
    
    Parameters:
    - dataset_path: Path to the Maestro dataset.
    - dataset_size: Number of MIDI files to load.
    

    """
    ii = 0
    all_encodings = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                if ii > dataset_size:
                    break
                midi_file = os.path.join(root, file)
                encoding = get_encodings(midi_file)
                all_encodings.append(encoding)
                ii += 1
                print("collecting song: ", ii)
                
    
    return all_encodings


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
  
# data loading
def get_batch(all_encodings, batch_size, block_size, dataset_size, train_test='train'):
    #loop through number of batches and get random subjects ts that is block_size long
    batch = []
    target_ts_batch = []
    for i in range(batch_size):
        if train_test == 'train': # first half of the dataset is for training and second half is for testing
            #get random song
            song = np.random.randint(0,dataset_size*0.75)
            ts = all_encodings[song]
        elif train_test == 'test':
            #get random song
            song = np.random.randint(dataset_size*0.75,dataset_size)
            ts = all_encodings[song]
        #get random block
        block = np.random.randint(0,len(ts)-(block_size+1))
        block_ts = ts[block:block+block_size] 
        target_ts = ts[block+1:block+block_size+1]
        #append to batch
        batch.append(block_ts)
        target_ts_batch.append(target_ts)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.int64)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.int64)
    return batch, target_ts_batch
    

@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, dataset_size):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        NLL_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data,batch_size, block_size, dataset_size, train_test=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, dropout, block_size,use_relational_position=True, max_relative_positions=25, use_relational_time_pitch=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.use_relational_position = use_relational_position
        if use_relational_position:
            #make max relative positions into a trainable parameter
            self.max_relative_positions = nn.Parameter(torch.tensor(max_relative_positions, dtype=torch.float32))
            # Relative position embeddings for K and V
            self.relative_k_positional_embeddings = nn.Embedding(2 * int(self.max_relative_positions.item()) + 1, head_size)
            self.relative_v_positional_embeddings = nn.Embedding(2 * int(self.max_relative_positions.item()) + 1, head_size)
        self.use_relational_time_pitch = use_relational_time_pitch
        if use_relational_time_pitch:
            #make max relative time and pitch into trainable parameters
            max_relative_time = 200
            max_relative_pitch = 128
            self.max_relative_time = nn.Parameter(torch.tensor(max_relative_time, dtype=torch.float32))
            self.max_relative_pitch = nn.Parameter(torch.tensor(max_relative_pitch, dtype=torch.float32))
            # Relative time embeddings for K and V
            self.relative_k_time_embeddings = nn.Embedding(2 * int(self.max_relative_time.item()) + 1, head_size)
            self.relative_v_time_embeddings = nn.Embedding(2 * int(self.max_relative_time.item()) + 1, head_size)
            # Relative pitch embeddings for K and V
            self.relative_k_pitch_embeddings = nn.Embedding(2 * int(self.max_relative_pitch.item()) + 1, head_size)
            self.relative_v_pitch_embeddings = nn.Embedding(2 * int(self.max_relative_pitch.item()) + 1, head_size)

    def forward(self, x, token_batch):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.use_relational_position:
            relative_key = self.relative_position(T,"k") # (T, T, C)
            #multiply relative_key with q
            # (T, T, C) @ (B, T, C) -> (B, T, T)
            attn_wei2 = torch.einsum('ijc,btc->btj', relative_key, q)
            wei += attn_wei2  
        if self.use_relational_time_pitch:
            #FIRST - add relative time embedding
            relative_key_time = self.relative_time(token_batch,"k") # (B, T, T, C)
            #multiply relative_key with q
            # (B, T, T, C) @ (B, T, C) -> (B, T, T)
            attn_wei2 = torch.einsum('bijc,btc->btj', relative_key_time, q)
            wei += attn_wei2  
            #SECOND - add relative pitch embedding
            relative_key_pitch = self.relative_pitch(token_batch,"k") # (B, T, T, C)
            #multiply relative_key with q
            # (B, T, T, C) @ (B, T, C) -> (B, T, T)
            attn_wei2 = torch.einsum('bijc,btc->btj', relative_key_pitch, q)
            wei += attn_wei2  
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        if self.use_relational_position:
            # Add relative V embeddings to attention output
            self.relative_value = self.relative_position(T,"v") # (T, T, C)
            # Perform the matrix multiplication
            # (T, T, C) @ (B, T, T) -> (B, T, C)
            relative_out = torch.einsum('ijc,btj->btc', self.relative_value, wei)
            out += relative_out
        if self.use_relational_time_pitch:
            #FIRST - add relative time embedding
            relative_value_time = self.relative_time(token_batch,"v") # (B, T, T, C)
            # Perform the matrix multiplication
            # (B, T, T, C) @ (B, T, T) -> (B, T, C)
            relative_time_out = torch.einsum('bijc,btj->btc', relative_value_time, wei)
            out += relative_time_out
            #SECOND - add relative pitch embedding
            relative_value_pitch = self.relative_pitch(token_batch,"v")
            # Perform the matrix multiplication
            # (B, T, T, C) @ (B, T, T) -> (B, T, C)
            relative_pitch_out = torch.einsum('bijc,btj->btc', relative_value_pitch, wei)
            out += relative_pitch_out
        
        return out
    
    def relative_position(self,T,embed):
        positions = torch.arange(T)
        position_distance = positions.unsqueeze(0)-positions.unsqueeze(1)
        position_distance_clamp = torch.clamp(position_distance, -int(self.max_relative_positions.item()), int(self.max_relative_positions.item()))
        # Add max_relative_positions to make it positive (because these are indices into the embeddings table, diagonal is still always equivalent to the zero distance embedding)
        position_distance_clamp_idx = position_distance_clamp + int(self.max_relative_positions.item())
        position_distance_clamp_idx = torch.LongTensor(position_distance_clamp_idx)
        if embed == "v":
            relative_embeddings = self.relative_v_positional_embeddings(position_distance_clamp_idx)
        elif embed == "k":
            relative_embeddings = self.relative_k_positional_embeddings(position_distance_clamp_idx)

        return relative_embeddings

    def relative_time(self,token_batch, embed):
        #this should read the times off of the incoming midi data, apply the time to the relevant frames(note and velocity) and calculate their relative distances
        #tokens are organized into (time,note,velocity) concatenated sequences 
       
        '''
        #time distances are different for each batch, must calculate per batch
        for batch in range(B):
            tokens = token_batch[batch]
            #find time tokens
            idx = tokens >= TIME_SHIFT_OFFSET
            time_token_values = tokens[idx]-TIME_SHIFT_OFFSET
            #change time derivatives into absolute time
            absolute_time = torch.cumsum(time_token_values, dim=0)+1 #this is so that notes that are played at the same time (zero time offset/derivative) are note seen as irrelevant (zero indices)[if they are at the beginning of a sequence]
            # Initialize an array to hold the propagated time values
            propagated_time_values = torch.zeros_like(tokens)
            # Propagate the time values by applying the time values for each subsequent note and velocity token
            current_time_value = 0
            time_index = 0
            for i in range(len(tokens)):
                if idx[i]:
                    current_time_value = absolute_time[time_index]
                    time_index += 1
                propagated_time_values[i] = np.round(current_time_value/10) #divide time by ten to make the increments more managable
            irrelevant_indices = propagated_time_values == 0
            time_distance = propagated_time_values.unsqueeze(0)-propagated_time_values.unsqueeze(1)
            time_distance_clamp = torch.clamp(time_distance, -self.max_relative_time, self.max_relative_time)
            # Add max_relative_time to make it positive (because these are indices into the embeddings table, diagonal is still always equivalent to the zero distance embedding)
            time_distance_clamp_idx = time_distance_clamp+self.max_relative_time
            #zero index for tokens with unknown notes
            time_distance_clamp_idx[:,irrelevant_indices] = 0 
            time_distance_clamp_idx[irrelevant_indices,:] = 0
            time_distance_clamp_idx = torch.LongTensor(time_distance_clamp_idx)
            #add to batch of relative time distances
            batch_time_distance_clamp_idx[batch] = time_distance_clamp_idx

                '''
        B,T = token_batch.shape

        # Find time tokens for the entire batch
        idx = token_batch >= TIME_SHIFT_OFFSET

        # Fill the padded tensor with time token values
        time_token_values = torch.zeros_like(token_batch)
        time_token_values[idx] = token_batch[idx] - TIME_SHIFT_OFFSET

        # Change time derivatives into absolute time for the entire batch
        absolute_time = torch.cumsum(time_token_values * idx, dim=1) + 1
        absolute_time = absolute_time.float()

        # Initialize an array to hold the propagated time values for the entire batch
        propagated_time_values = torch.zeros_like(token_batch, dtype=torch.float)

        # Propagate the time values by applying the time values for each subsequent note and velocity token
        current_time_values = torch.full((B,), float('nan'), dtype=torch.float) # Initialize to NaN to apply nan to tokens with unknown time (given current sequence block)

        for i in range(T):
            current_time_values[idx[:, i]] = absolute_time[idx[:, i], i]
            propagated_time_values[:, i] = torch.round(current_time_values / 10)


        time_distance = propagated_time_values.unsqueeze(1) - propagated_time_values.unsqueeze(2)

        # Zero index for tokens with unknown times
        time_distance_clamp = torch.clamp(time_distance, -int(self.max_relative_time.item()), int(self.max_relative_time.item()))

        # Add max_relative_time to make it positive
        time_distance_clamp_idx = time_distance_clamp + int(self.max_relative_time.item())

        unknown_time = torch.isnan(time_distance_clamp_idx)
        time_distance_clamp_idx[unknown_time] = 0
        # Convert to LongTensor and assign to batch index
        batch_time_distance_clamp_idx = torch.LongTensor(time_distance_clamp_idx.to(torch.long))


        if embed == "v":
            relative_embeddings = self.relative_v_time_embeddings(batch_time_distance_clamp_idx)
        elif embed == "k":
            relative_embeddings = self.relative_k_time_embeddings(batch_time_distance_clamp_idx)

        return relative_embeddings 
    
    def relative_pitch(self,token_batch,embed):
        #I believe this should help the model learn the current key and scale being used for a song
        #this should read the notes off of the incoming midi data, apply the note to the relevant frames(time and velocity) and calculate their relative distances
        #tokens are organized into (time,note,velocity) concatenated sequences 
        
        '''
        #time distances are different for each batch, must calculate per batch
        for batch in range(B):
            tokens = token_batch[batch]
            #find note tokens
            idx = tokens < VELOCITY_OFFSET #any tokens below velocity offset are note on or note off tokens
            token_values = tokens[idx]
            ind = token_values >= NOTE_OFF_OFFSET #find note off tokens
            token_values[ind] = token_values[ind]-NOTE_OFF_OFFSET #get absolute note values
            token_values += 1 #add 1 so that the first valid index is 1, and 0 can correspond to unknown note indices (time points where we don't have note info) so the embedding can learn 
            # Initialize an array to hold the propagated time values
            propagated_note_values = torch.zeros_like(tokens)
            # Propagate the note values by applying the note values to the relevant time and token indices (sequences are arranged as [time, note, velocity] therefore we need to apply note value on either side of each index
            current_note_value = 0  # if you do not start a sequence with a note, then the note for the first element of the sequence is unknown (represented here as embedding index 0)
            time_index = 0
            for i in range(len(tokens)):
                if idx[i]:
                    current_note_value = token_values[time_index]
                    time_index += 1
                if i > 0 and i < len(tokens) - 1: #if there are zeros on both sides of index
                    propagated_note_values[i - 1] = current_note_value
                    propagated_note_values[i] = current_note_value
                    propagated_note_values[i + 1] = current_note_value
                elif i > 0 and i == len(tokens) - 1: #if there is a zero on the left side of index
                    propagated_note_values[i - 1] = current_note_value
                    propagated_note_values[i] = current_note_value
                elif i == 0: #if there is a zero on the right side of the index
                    propagated_note_values[i] = current_note_value
                    propagated_note_values[i + 1] = current_note_value

            irrelevant_indices = propagated_note_values == 0
            note_distance = propagated_note_values.unsqueeze(0)-propagated_note_values.unsqueeze(1)
            note_distance_clamp = torch.clamp(note_distance, -self.max_relative_pitch, self.max_relative_pitch)
            # Add max_relative_pitch to make it positive (because these are indices into the embeddings table, diagonal is still always equivalent to the zero distance embedding)
            note_distance_clamp_idx = note_distance_clamp+self.max_relative_pitch
            #zero index for tokens with unknown notes
            note_distance_clamp_idx[:,irrelevant_indices] = 0 
            note_distance_clamp_idx[irrelevant_indices,:] = 0
            note_distance_clamp_idx = torch.LongTensor(note_distance_clamp_idx)
            #add to batch of relative note distances
            batch_note_distance_clamp_idx[batch] = note_distance_clamp_idx
        '''
        B,T = token_batch.shape
        # Find note tokens for the entire batch
        idx = token_batch < VELOCITY_OFFSET  # Any tokens below velocity offset are note on or note off tokens
        token_values = token_batch.clone()
        token_values[~idx] = 0  # Mask out non-note tokens

        # Find note off tokens and get absolute note values
        ind = token_values >= NOTE_OFF_OFFSET
        token_values[ind] = token_values[ind] - NOTE_OFF_OFFSET  # Get absolute note values
        token_values[idx] += 1  # Add 1 so that the first valid index is 1
        token_values = token_values.float()

        # Initialize an array to hold the propagated note values for the entire batch
        propagated_note_values = torch.zeros_like(token_batch, dtype=torch.float)

        # Propagate the note values
        current_note_values = torch.full((B,), float('nan'), dtype=torch.float) # Initialize to NaN to apply nan to tokens with unknown time (given current sequence block)

        for i in range(T):
            current_note_values[idx[:, i]] = token_values[idx[:, i],i]
            
            if i > 0 and i < T - 1:
                propagated_note_values[:, i - 1] = current_note_values
                propagated_note_values[:, i] = current_note_values
                propagated_note_values[:, i + 1] = current_note_values
            elif i > 0 and i == T - 1:
                propagated_note_values[:, i - 1] = current_note_values
                propagated_note_values[:, i] = current_note_values
            elif i == 0:
                propagated_note_values[:, i] = current_note_values
                propagated_note_values[:, i + 1] = current_note_values


        note_distance = propagated_note_values.unsqueeze(1) - propagated_note_values.unsqueeze(2)
        note_distance_clamp = torch.clamp(note_distance, -int(self.max_relative_pitch.item()), int(self.max_relative_pitch.item()))

        # Add max_relative_pitch to make it positive
        note_distance_clamp_idx = note_distance_clamp + int(self.max_relative_pitch.item())
        unknown_notes = torch.isnan(note_distance_clamp_idx)
        note_distance_clamp_idx[unknown_notes] = 0
        # Convert to LongTensor
        batch_note_distance_clamp_idx = torch.LongTensor(note_distance_clamp_idx.to(torch.long))

        if embed == "v":
            relative_embeddings = self.relative_v_pitch_embeddings(batch_note_distance_clamp_idx)
        elif embed == "k":
            relative_embeddings = self.relative_k_pitch_embeddings(batch_note_distance_clamp_idx)

        return relative_embeddings
  
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size,use_relational_position=True,use_relational_time_pitch=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, block_size,use_relational_position=use_relational_position,use_relational_time_pitch=use_relational_time_pitch) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tokens):
        out = torch.cat([h(x,tokens) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd), #to match the archetecture of the transformer used in MUSIC TRANSFORMER from Google Brain
            nn.ReLU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size,use_relational_position=True,use_relational_time_pitch=True):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size,use_relational_position=use_relational_position,use_relational_time_pitch=use_relational_time_pitch)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, tokens):
        x = x + self.sa(self.ln1(x),tokens) #residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class CustomSequential(nn.Sequential):
    def forward(self, x1, tokens_batch):
        for module in self:
            x1 = module(x1, tokens_batch)
        return x1, tokens_batch #pass the tokens through the layers without any transformation (so that they can be used in relational time/pitch embeddings in later layers)
    
# super simple bigram model
class Music_transformer_Model(nn.Module):

    def __init__(self,n_embd, n_head, dropout, vocab_size, block_size, n_layer, device,use_relational_position=True,use_relational_time_pitch=True,use_positional_encoding = True):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
        self.blocks = CustomSequential(*[Block(n_embd, n_head, dropout, block_size,use_relational_position=use_relational_position,use_relational_time_pitch=use_relational_time_pitch) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.use_positional_encoding = use_positional_encoding

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        if self.use_positional_encoding:
            pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
        else:
            x = tok_emb
        x, idx = self.blocks(x,idx) # (B,T,C) also gets the tokens themselves (for relational time and pitch embedding)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-self.block_size:]
            #print("idx cond shape: ",idx_cond.shape)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
