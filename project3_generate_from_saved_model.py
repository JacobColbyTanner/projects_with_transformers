import torch
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model3 import get_batch, estimate_loss, MUSIC_transformer_model, load_maestro_dataset, midi_to_tensor, tensor_to_midi, CombinedMSEAndL1Loss
from transformers import GPT2Tokenizer
import os
import mido
import matplotlib.pyplot as plt
import numpy as np

###MODEL hyperparameters###

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 50 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20 #number of iterations to evaluate the loss
plot_eval_output = False
num_notes = 128 #This is the number of notes on the MIDI keyboard
n_embd = num_notes
n_head = 6
n_layer = 4
dropout = 0.1 #dropout probability
time_resolution =  96*2 #try multiples of 96 (its easier to train the model to vary its outputs with a higher time_resolution)
dataset_size = 100 #number of MIDI files to load
loss_type = 'BCE' #'MSE' or 'BCE' (Binary Cross Entropy loss with logits)   
# Note that changing this parameter slightly varies the architecture of the model
# additionally, with BCE loss, the model is trained for binary note presses, and therefore
# ignores the velocity of the note press




### GENERATE HYPERPARAMETERS
threshold = 0.1
generated_song_length = 1500
remove_original_song_segment_from_generated = True

# Path to the Maestro dataset
dataset_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/'

# Load the dataset
midi_tensors = load_maestro_dataset(dataset_path, dataset_size = dataset_size, loss_type = loss_type)


# Load the model
model_path = 'saved_models/MUSIC_transformer_model.pth'
model = MUSIC_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device, loss_type)
model.load_state_dict(torch.load(model_path))
model.eval()

# Switch to evaluation mode
model.eval()
# generate from the model

batch, target_ts_batch = get_batch(midi_tensors,batch_size, block_size, dataset_size, train_test='train')
context = batch[0,:,:].squeeze(0)
predicted_MIDI = model.generate(context, max_new_tokens=generated_song_length,threshold=threshold)

print(predicted_MIDI.shape)

if remove_original_song_segment_from_generated:
    predicted_MIDI = predicted_MIDI[block_size:]

# Add zeros to the end of axis 0, this enables tensor to midi to see note off events
zeros = torch.zeros((50, predicted_MIDI.shape[1]))
predicted_MIDI = torch.cat((predicted_MIDI, zeros), axis=0)


# Save the generated MIDI tensor to a MIDI file
output_path = '/Users/jacobtanner/Downloads/transformer_generated_song5.midi'
tensor_to_midi(predicted_MIDI.T, output_path, time_resolution=time_resolution,loss_type=loss_type)

#plot the predicted fMRI data
plt.figure()
plt.imshow(predicted_MIDI.detach().numpy().T)
plt.title("Generated MIDI tensor")
plt.ylabel("Note")
plt.xlabel("Time step")
plt.show()