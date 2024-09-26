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



train_type = 'many' # 'once' or 'many'. 
# Many will train the model many times for different iterations and
# show plots suggesting how the model improves its behavior using 
# the attention weights with more iterations

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 1000
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




if train_type == 'once':

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
    predicted_fMRI = m.generate(context, max_new_tokens=1000)

    #plot the predicted fMRI data
    plt.plot(predicted_fMRI.detach().numpy())
    plt.draw()
    plt.pause(0.1)


    # Access attention weights from each head in the first block
    attention_weights = []
    for head in model.blocks[0].sa.heads:
        attention_weights.append(head.attention_weights.detach().cpu().numpy().squeeze())

    # Print or plot the attention weights
    for i, weights in enumerate(attention_weights):
        
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights for Head {i}')
        plt.draw()
        plt.pause(0.1)

    # calculate how long it takes for the model predictions to explode
    mean_abs_activity = np.mean(np.abs(predicted_fMRI.detach().numpy()),axis=1)
    plt.figure(figsize=(10, 8))
    plt.plot(mean_abs_activity)
    plt.title('Where does prediction explode?')
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Activity')
    plt.draw()
    plt.pause(0.1)
    

    

elif train_type == 'many':

    max_iters_select = np.arange(100,2700,500)
    print("Iterations that will be used for training each model: ", max_iters_select)
    attention_mean = []
    where_activity_explodes = []

    for select in range(len(max_iters_select)):
        max_iters = max_iters_select[select]
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
        predicted_fMRI = m.generate(context, max_new_tokens=10000)

        #get mean attention weights
         # Access attention weights from each head in the first block
        attention_weights_m = []
        for head in model.blocks[0].sa.heads:
            attention_weights_m.append(np.mean(head.attention_weights.detach().cpu().numpy().squeeze(),axis=0))
        attention_mean.append(np.mean(attention_weights_m,axis=0))

        #get mean absolute activity to check where the predictions begin to explode (predictions are no longer good as the activity grows increasingly large (positive or negative))
        mean_abs_activity = np.mean(np.abs(predicted_fMRI.detach().numpy()),axis=1)
        #take estimate index of where activity explodes and subtract block_size to get where new predictions are no longer good
        idx = mean_abs_activity>10
        try:
            print("this is where")
            idx = np.where(idx)[0][0]
            where_activity_explodes.append(idx-block_size)
        except:
            where_activity_explodes.append(10000)

    #plot the mean attention weights
    # label each line with the number of iterations used for that model
    plt.figure(figsize=(10, 8))
    for i, attention in enumerate(attention_mean):
        plt.plot(attention, label=f'{max_iters_select[i]} iterations')
    plt.legend()
    plt.title('Mean Attention Weights')
    plt.xlabel('Time points attended to')
    plt.ylabel('Mean Attention Weight')
    plt.draw()
    plt.pause(0.1)

    #plot where the predictions explode
    # label each line with the number of iterations used for that model
    plt.figure(figsize=(10, 8)) 
    plt.plot(np.array(where_activity_explodes))
    plt.title('Where do predictions explode as you increase iterations?')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('Time where predictions explode')
    plt.show()

