import torch
import torch.nn as nn
from torch.nn import functional as F
from models.transformer_model2 import estimate_loss, get_fMRI_batch, fMRI_transformer_model, ActivationCollector,get_within_vs_between_embeddings, get_within_vs_between_facts
from transformers import GPT2Tokenizer
import numpy as np
from scipy.io import loadmat, savemat
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
import os


train_type = 'tracked' # 'normal' or 'tracked'. 
# Tracked will track certain variables across training the model and
# show plots suggesting how the model improves its behavior using 
# the attention weights with more iterations
save_plots = True
#if True this will save the plots and also the embedding trajectories 
# for the trained model (only under the "tracked" condition above)
# this will also save the data used for the plots
embedding_save_path = 'data/fMRI/embedding_trajectories_training_steps.npy'
sample_from_test = False
save_model = False #if true, then the model will be saved after training (only in the tracked condition)
model_save_path = 'saved_models/fMRI/fMRI_transformer_model_200000_training_steps.pth'

# hyperparameters
use_pretrained_model = False
model_load_path = 'saved_models/fMRI/fMRI_transformer_model_100000_training_steps.pth'
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 22 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
num_brain_regions = 200 #number of brain regions in the fMRI data, also making this the embedding dimension
n_embd = num_brain_regions
n_head = 4
n_layer = 1
dropout = 0.1 #dropout probability
split_type = 'subject'
predict_with = 'train' #predict with 'train' or 'test' data
add_first_and_final_frame = False #whether or not to keep the first (observed) and final (predicted) frame of the fMRI data in the embedding trajectory
# ------------

torch.manual_seed(1337)


#load the fMRI data
# open data
data = loadmat('/Users/jacobtanner/Documents/schaefer200_structfunc_data_36pSpike.mat')
data = data['sf_hcp_data']




if train_type == 'normal':

    model = fMRI_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data, eval_iters, block_size, batch_size,split_type=split_type)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")


        # sample a batch of data
        batch, target_ts_batch = get_fMRI_batch(data,batch_size, block_size, train_test='train',split_type=split_type)

        # evaluate the loss
        logits, loss = model(batch, target_ts_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    batch_size_for_predict = 1
    block_size_for_predict = 1000
    batch, target_ts_batch = get_fMRI_batch(data,batch_size_for_predict, block_size_for_predict, train_test='train')
    context = batch[0,0:block_size,:].squeeze(0)
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
    

    

elif train_type == 'tracked':

    
    record_interval = 200
    attention_mean = []
    where_activity_explodes = []
    all_corr = []
    mse_ts = []
    all_sub_similarity_diff = []
    all_sub_fact_similarity_diff = []
    train_loss_all = []
    test_loss_all = []

    model = fMRI_transformer_model(n_embd, n_head, dropout, block_size, n_layer, device)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    if use_pretrained_model:
        model.load_state_dict(torch.load(model_load_path))
        print("Loaded pretrained model")
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, data, eval_iters, block_size, batch_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
            #store training loss
            train_loss_all.append(losses['train'])
            test_loss_all.append(losses['test'])

        # sample a batch of data
        batch, target_ts_batch = get_fMRI_batch(data,batch_size, block_size, train_test='train')

        # evaluate the loss
        logits, loss = model(batch, target_ts_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % record_interval == 0 or iter == max_iters - 1:
            model.eval()
            print("recording info...")
            # generate from the model
            batch_size_for_predict = 1
            block_size_for_predict = 1000+block_size
            batch, target_ts_batch = get_fMRI_batch(data,batch_size_for_predict, block_size_for_predict, train_test=predict_with)
            context = batch[0,0:block_size,:].squeeze(0)
            predicted_fMRI = m.generate(context, max_new_tokens=4000)
            # Calculate the instantaneous MSE
            time_series1 = predicted_fMRI[block_size:1000+block_size].detach().numpy()
            time_series2 = batch[0,block_size:1000+block_size,:].squeeze(0).detach().numpy()
            mse_ts.append(np.mean((time_series1 - time_series2) ** 2, axis=1))
            print("MSE: ", np.mean(mse_ts[-1]))
            predicted_FC = np.corrcoef(predicted_fMRI[block_size:1000+block_size].detach().numpy().T)
            #actual FC
            actual_FC = np.corrcoef(batch[0,block_size:1000+block_size,:].squeeze(0).T)
            #correlate predicted and actual FC
            corr = np.corrcoef(predicted_FC.flatten(),actual_FC.flatten())[0,1]
            print("Correlation between predicted and actual FC: ", corr)
            all_corr.append(corr)

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
                
                idx = np.where(idx)[0][0]
                print("Activity explodes at time point: ", idx-block_size)
                where_activity_explodes.append(idx-block_size)
            except:
                print("activity does not explode within 4000 time steps")
                where_activity_explodes.append(4000)



            num_subjects = 5
            number_of_prediction_steps = 3
            all_facts = np.zeros((num_subjects,4,number_of_prediction_steps,num_brain_regions))
            if add_first_and_final_frame:
                sample_embedding_trajectories = np.zeros((num_subjects,4,number_of_prediction_steps*8,num_brain_regions))
            else:
                sample_embedding_trajectories = np.zeros((num_subjects,4,number_of_prediction_steps*6,num_brain_regions))
            if sample_from_test:
                subjects_start = 76 #this is where the train test split occurs, so this is the first test subject
            else:
                subjects_start = 0 #this will sample from subjects that were in the training data

            for SS in range(num_subjects):
                subject = subjects_start+SS
                for scan in range(4):
                    collect_embeddings = ActivationCollector(model,data,block_size,number_of_prediction_steps,subject,scan,add_first_and_final_frame)
                    sample_embedding_trajectories[SS,scan,:,:] = collect_embeddings.all_embedding_trajectories
                    all_facts[SS,scan,:,:] = collect_embeddings.all_facts
            r_within, r_between_flat, diff = get_within_vs_between_embeddings(sample_embedding_trajectories,num_subjects,number_of_prediction_steps,add_first_and_final_frame)
            print("Difference between within and between subject similarity: ", diff)
            all_sub_similarity_diff.append(diff)
            
            r_within, r_between_flat, diff = get_within_vs_between_facts(all_facts,num_subjects)
            print("Difference between within and between subject similarity for facts: ", diff)
            all_sub_fact_similarity_diff.append(diff)
            
            



            model.train()

    print("Collecting all embedding trajectories...")
    number_of_prediction_steps = np.round(1100//block_size) #total number of non-overlapping blocks available per time series
    all_embedding_trajectories = np.zeros((95,4,number_of_prediction_steps*8,num_brain_regions))
    start_time = time.time()
    for subject in range(95):
        print("--------Subject: ", subject)
        for scan in range(4):
            print("Scan: ", scan)
            
            
            #run class to collect embedding transformations for a select subject/scan and a selected number of prediction/generation steps with the model
            #subject = 0
            #scan = 0
            
            collect_embeddings = ActivationCollector(model,data,block_size,number_of_prediction_steps,subject,scan,add_first_and_final_frame=True)
            all_embedding_trajectories[subject,scan,:,:] = collect_embeddings.all_embedding_trajectories
    total_time = time.time()-start_time 

    print("Total time to collect all embedding trajectories in seconds: ", np.round(total_time))
    
    # Ensure the directory exists
    output_dir = 'data/fMRI'
    os.makedirs(output_dir, exist_ok=True)
    # Ensure the directory exists
    output_dir2 = 'saved_models/fMRI'
    os.makedirs(output_dir2, exist_ok=True)

    #save the model
    if save_model:
        torch.save(model.state_dict(), model_save_path)

    if save_plots:
        #save the embedding trajectories as numpy array
        np.save(embedding_save_path,all_embedding_trajectories)

    max_iters_select = np.arange(0,max_iters+record_interval,record_interval)  #[200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600]
    #plot the mean attention weights
    # label each line with the number of iterations used for that model
    plt.figure(figsize=(10, 8))
    for i, attention in enumerate(attention_mean):
        plt.plot(attention, label=f'{max_iters_select[i]} iterations')
    plt.legend()
    plt.title('Mean Attention Weights')
    plt.xlabel('Time points attended to')
    plt.ylabel('Mean Attention Weight')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/mean_attention_weights.png')
        np.save('data/fMRI/mean_attention_weights.npy',attention_mean)
    plt.draw()
    plt.pause(0.1)

    plt.figure(figsize=(10, 8))
    num_peaks = []
    for i, attention in enumerate(attention_mean):
        # Find peaks
        peaks, _ = find_peaks(attention)
        # Number of peaks
        num_peaks.append(len(peaks))
    plt.plot(np.array(num_peaks))
    plt.title('Number of Peaks in Attention Weights')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('Number of Peaks')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/num_peaks_attention_weights.png')
        np.save('data/fMRI/num_peaks_attention_weights.npy',np.array(num_peaks))
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
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/predictions_explode.png')
        np.save('data/fMRI/predictions_explode.npy',np.array(where_activity_explodes))
    plt.draw()
    plt.pause(0.1)

    #plot training and test loss
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_all, label='Train Loss')
    plt.plot(test_loss_all, label='Test Loss')
    #change y range to zoom in to range between 0 and 0.1   
    plt.ylim(0,0.002)
    plt.legend()
    plt.title('Training and Test Loss')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('Loss')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/training_and_test_loss.png')
        np.save('data/fMRI/training_loss.npy',np.array(train_loss_all))
        np.save('data/fMRI/test_loss.npy',np.array(test_loss_all))
    plt.draw()
    plt.pause(0.1)
    
    #plot instantaneous MSE
    plt.figure(figsize=(10, 8))
    for i, mse in enumerate(mse_ts):
        plt.plot(mse, label=f'{max_iters_select[i]} iterations')
    plt.legend()
    plt.title('Instantaneous MSE')
    plt.xlabel('Time points (pred and obs)')
    plt.ylabel('Instantaneous MSE')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/instantaneous_MSE.png')
        np.save('data/fMRI/instantaneous_MSE.npy',np.array(mse_ts))
    plt.draw()
    plt.pause(0.1)

    #plot MSE (average across time points) between real and generated fMRI data (from the same initial time step)
    plt.figure(figsize=(10, 8))
    mse_all = []
    for i, mse in enumerate(mse_ts):
        mse_all.append(np.mean(mse))
    plt.plot(mse_all)
    plt.legend()
    plt.title('MSE: Predicted vs. Actual fMRI data (predicted 1000 time steps)')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('Instantaneous MSE')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/MSE_training_steps.png')
    plt.draw()
    plt.pause(0.1)
    
    #plot instantaneous MSE
    plt.figure(figsize=(10, 8))
    plt.plot(mse_ts[-1], label=f'{max_iters_select[-1]} iterations')
    plt.legend()
    plt.title('Instantaneous MSE for final training step')
    plt.xlabel('Time points (pred and obs)')
    plt.ylabel('Instantaneous MSE')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/instantaneous_MSE_final_step.png')
    plt.draw()
    plt.pause(0.1)

    #plot both predicted and observed fMRI data
    plt.figure(figsize=(10, 8))
    plt.plot(time_series1[0:50,1], label='Predicted')
    plt.plot(time_series2[0:50,1], label='Observed')
    plt.legend()
    plt.title('Predicted vs. Observed fMRI data')
    plt.xlabel('Time points')
    plt.ylabel('Activity')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/predicted_vs_observed_fMRI.png')
        np.save('data/fMRI/predicted_fMRI.npy',time_series1)
        np.save('data/fMRI/observed_fMRI.npy',time_series2)
    plt.draw()
    plt.pause(0.1)

    #plot sub similarity (within-between) difference of embedding space trajectories
    plt.figure(figsize=(10, 8))
    plt.plot(all_sub_similarity_diff)
    plt.legend()
    plt.title('subject similarity difference in embedding space')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('within sub corr - between sub corr')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/within_vs_between_corr_embedding.png')
        np.save('data/fMRI/within_vs_between_corr_embedding.npy',all_sub_similarity_diff)
    plt.draw()
    plt.pause(0.1)
    

    #plot the correlation between predicted and actual FC
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(all_corr))
    plt.title('Correlation between predicted and actual FC')
    plt.xlabel('Number of iterations of training')
    #change x ticks to correspond to num iterations
    plt.xticks(np.arange(len(max_iters_select)),max_iters_select)
    plt.ylabel('Correlation (aFC vs. pFC)')
    #save the plot
    if save_plots:
        plt.savefig('figures/fMRI/correlation_pred_vs_act_FC.png')
        np.save('data/fMRI/correlation_pred_vs_act_FC.npy',np.array(all_corr))
    plt.show()


