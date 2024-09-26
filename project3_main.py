import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model3 import get_batch, estimate_loss, MUSIC_transformer_model, load_maestro_dataset, midi_to_tensor, tensor_to_midi, CombinedMSEAndL1Loss
from transformers import GPT2Tokenizer
import os
import mido
import matplotlib.pyplot as plt
import numpy as np


#option to train starting with a pretrained model
train_from_pretrained = True

# hyperparameters
batch_size = 100 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20 #number of iterations to evaluate the loss
plot_eval_output = False
num_notes = 128 #This is the number of notes on the MIDI keyboard
n_embd = num_notes
n_head = 8
n_layer = 4
dropout = 0.25 #dropout probability
dataset_size = 1000 #number of MIDI files to load
time_resolution =  96*2 #try multiples of 96 (its easier to train the model to vary its outputs with a higher time_resolution)
loss_type = 'BCE' #'MSE' or 'BCE' (Binary Cross Entropy loss with logits)   
# Note that changing this parameter slightly varies the architecture of the model
# additionally, with BCE loss, the model is trained for binary note presses, and therefore
# ignores the velocity of the note press

# ------------

torch.manual_seed(1337)


# Path to the Maestro dataset
dataset_path = '/Users/jacobtanner/Downloads/maestro-v1.0.0/'

# Load the dataset
midi_tensors = load_maestro_dataset(dataset_path, dataset_size = dataset_size, time_resolution = time_resolution, loss_type = loss_type)

# Example: Print the shape of the first tensor
if midi_tensors:
    print(midi_tensors[0].shape)
    plt.imshow(midi_tensors[0].numpy())
    plt.title("Example midi tensor")
    plt.ylabel("Note")
    plt.xlabel("Time step")
    plt.colorbar()
    #plt.draw()
    #plt.pause(0.1)
    plt.show()
else:
    print("No MIDI files found in the dataset.")



model = MUSIC_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device, loss_type)
model = model.to(device)

if train_from_pretrained:
    model.load_state_dict(torch.load('saved_models/MUSIC_transformer_model_ver2.pth'))

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses, output = estimate_loss(model, midi_tensors, eval_iters, block_size, batch_size, dataset_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
        #print mean output
        print("Mean output: ", torch.mean(output))
        #plot the predicted fMRI data
        if plot_eval_output:
            plt.figure()
            plt.imshow(output[0].detach().numpy().squeeze())
            plt.title("Predicted MIDI tensor (during training)")
            plt.ylabel("Time Step")
            plt.xlabel("Note")
            plt.draw()
            plt.pause(0.1)

    # sample a batch of data
    xb, yb = get_batch(midi_tensors, batch_size, block_size, dataset_size, train_test='train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Switch to evaluation mode
model.eval()
# generate from the model
generated_song_length = 10
batch, target_ts_batch = get_batch(midi_tensors,batch_size, block_size, dataset_size, train_test='train')
context = batch[0,:,:].squeeze(0)
predicted_MIDI = model.generate(context, max_new_tokens=generated_song_length)

print(predicted_MIDI.shape)

# Save the generated MIDI tensor to a MIDI file
output_path = '/Users/jacobtanner/Downloads/transformer_generated_song.midi'
tensor_to_midi(predicted_MIDI.T, output_path, time_resolution = time_resolution, loss_type=loss_type)

#plot the predicted fMRI data
plt.figure()
plt.imshow(predicted_MIDI.detach().numpy().T)
plt.title("Generated MIDI tensor")
plt.ylabel("Note")
plt.xlabel("Time step")
plt.show()

#save the model
torch.save(model.state_dict(), 'saved_models/MUSIC_transformer_model_ver2.pth')