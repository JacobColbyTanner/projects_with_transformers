import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model4 import decode_velocity, get_batch, estimate_loss, Music_transformer_Model, load_maestro_dataset, get_decodings, get_encodings
import numpy as np
from models.LSTM_model import LSTMModel

###MODEL hyperparameters###


# hyperparameters
batch_size = 25 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
max_iters = 2500
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 200
n_layer = 1
hidden_size = 500
dataset_size = 10
# ------------
output_midi_file_path = '/Users/jacobtanner/Downloads/LSTM_piano_music2.midi'



# Path to the Maestro dataset
midi_file_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/2017/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.midi'

# Load the dataset and translate a song into a list of tokens
encodings = get_encodings(midi_file_path)

# Get the vocab size
vocab_size = 1023

model = LSTMModel(vocab_size, n_embd, hidden_size, num_layers=n_layer)
model = model.to(device)
# Load the model
model_path = 'saved_models/model4_music_transformer_yes_relative_positional_embedding_ver2.pth'
model.load_state_dict(torch.load(model_path))

# Switch to evaluation mode
model.eval()
# generate from the model
context = torch.tensor(encodings[0]).unsqueeze(0).unsqueeze(0)
print("context shape: ",context.shape)
pred_encodings = model.generate(context, max_new_tokens=5000)[0].tolist()
get_decodings(pred_encodings, output_midi_file_path)

print("Generated song saved to: ", output_midi_file_path)