
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
# Original version of this code comes from a transformer building tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe


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
  


def get_fMRI_batch(data,batch_size, block_size, train_test='train',split_type = 'subject'):
    
    #loop through number of batches and get random subjects ts that is block_size long
    for i in range(batch_size):
        if split_type == 'subject':
            if train_test == 'train':
                #get random subject
                subject = np.random.randint(0,76)
                subject_ts_across_scans = data[0,subject]
                #get random scan
                scan = np.random.randint(0,4)
            elif train_test == 'test':
                #get random subject
                subject = np.random.randint(76,95)
                subject_ts_across_scans = data[0,subject]
                #get random scan
                scan = np.random.randint(0,4)
            
        elif split_type == 'scan':
            #get random subject
            subject = np.random.randint(0,95)
            subject_ts_across_scans = data[0,subject]
            #get random scan, first two scans are train, last 2 are test
            if train_test == 'train':
                scan = np.random.randint(0,2)
            elif train_test == 'test':
                scan = np.random.randint(2,4)
        
        ts = subject_ts_across_scans['func']['scan'][0][0]['ts'][0][scan]
        #get random block
        block = np.random.randint(0,ts.shape[0]-(block_size+1))
        block_ts = ts[block:block+block_size,:]
        target_ts = ts[block+1:block+block_size+1,:]
        #append to batch
        if i == 0:
            batch = np.expand_dims(block_ts, axis=0)
            target_ts_batch = np.expand_dims(target_ts, axis=0)
        else:
            batch = np.concatenate((batch,np.expand_dims(block_ts, axis=0)),axis=0)
            target_ts_batch = np.concatenate((target_ts_batch,np.expand_dims(target_ts, axis=0)),axis=0)
    #convert to tensor
    batch = torch.tensor(batch, dtype=torch.float)
    target_ts_batch = torch.tensor(target_ts_batch, dtype=torch.float)
    return batch, target_ts_batch



@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size,split_type='subject'):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_fMRI_batch(data,batch_size, block_size,train_test=split,split_type=split_type)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
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
        self.proj = nn.Linear(n_embd, n_embd)
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
        head_size = n_embd // n_head
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
class fMRI_transformer_model(nn.Module):

    def __init__(self,n_embd, n_head, dropout, block_size, n_layer, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.device = device
        self.attention_weights = []

    def forward(self, idx, targets=None):
        B, T, C = idx.shape

        
        fmri_embedding = idx # fmri data is already in a good embedding space
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = fmri_embedding + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        

        if targets is None:
            loss = None
        else:
            #perform MSE loss between predicted and actual time series in shape (B,T,C)
            loss = F.mse_loss(x, targets)
      

        return x, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            #size of input is block_size x num_brain_regions
            idx_cond = idx[-self.block_size:,:].unsqueeze(0)
            # get the predictions
            next_pred, loss = self(idx_cond)
            
            idx_next = next_pred[:,-1,:].squeeze(0).unsqueeze(0)
            #print("idx shape", idx.shape)
            #print("idx_next shape", idx_next.shape)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=0) # (block_size, num_brain_regions)
        return idx




class ActivationCollector:
    def __init__(self,model,data,block_size,number_of_prediction_steps,subject,scan,add_first_and_final_frame=True):
        self.model = model
        self.block_size = block_size
        self.activations = []
        self.heads_just_done = False
        self.head_outputs_all = []
        self.hook_handles = []
        self.head_num = 0
        self.num_steps = number_of_prediction_steps
        self.data = data
        self.register_hooks()
        self.all_embedding_trajectories, self.all_facts = self.generate_activations(subject,scan,add_first_and_final_frame)
        self.remove_hooks() #this saves memory

    def get_activation(self, name):
        def hook(model, input, output):
            # if name contains 'head', store the output
            #print("collecting activations of: ",name)
            if 'head' in name:
                #print("collecting head activations")
                self.heads_just_done = True
                head_output_next = output[:,-1,:].detach().numpy().squeeze().squeeze() #get last embedding from block size (because this is what becomes the prediction)
                if self.head_num == 0:
                    self.head_outputs_all = head_output_next
                    #print("collected first head")
                else:
                    self.head_outputs_all = np.concatenate((self.head_outputs_all,head_output_next),axis=0)
                self.head_num += 1
            else:
                if self.heads_just_done:
                    self.activations.append(self.head_outputs_all)
                    self.activations.append(output[:,-1,:].detach().numpy().squeeze().squeeze()) #get last embedding from block size (because this is what becomes the prediction)
                    self.heads_just_done = False
                else:
                    self.activations.append(output[:,-1,:].detach().numpy().squeeze().squeeze()) #get last embedding from block size (because this is what becomes the prediction)
        return hook
    

    def register_hooks(self):
        # Register hooks for the layers of interest and store the handles
        for i, head in enumerate(self.model.blocks[0].sa.heads):
            handle = head.register_forward_hook(self.get_activation(f'head_{i}'))
            self.hook_handles.append(handle)

        handle = self.model.blocks[0].sa.proj.register_forward_hook(self.get_activation('proj'))
        self.hook_handles.append(handle)

        handle = self.model.blocks[0].ln1.register_forward_hook(self.get_activation('ln_1'))
        self.hook_handles.append(handle)

        handle = self.model.blocks[0].ln2.register_forward_hook(self.get_activation('ln_2'))
        self.hook_handles.append(handle)

        handle = self.model.blocks[0].ffwd.net[2].register_forward_hook(self.get_activation('FF_out'))
        self.hook_handles.append(handle)

    def remove_hooks(self):
        # Remove all registered hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def get_subject_time_series(self,subject,scan,data):
        #get subject
        subject_ts_across_scans = data[0,subject]
        #select scan
        subject_ts = torch.tensor(subject_ts_across_scans['func']['scan'][0][0]['ts'][0][scan], dtype=torch.float)
        return subject_ts

    def generate_activations(self,subject,scan,add_first_and_final_frame):
        #select a subjects time series
        subject_ts = self.get_subject_time_series(subject,scan,self.data)
        start = 0
        stop = self.block_size
        all_facts = []
        for i in range(self.num_steps):
            #print("prediction step: ",i)
            context = subject_ts[start:stop,:] #get the context
            # generate from the model
            predicted_fMRI = self.model.generate(context, max_new_tokens=1)
            self.activations = np.array(self.activations)
            T, C = context.shape
            #get the positional embeddings
            pos_emb = self.model.position_embedding_table(torch.arange(T))
            #add to context to get the embeddings that are passed to the model
            context_pos = context + pos_emb
            #grab last frame corresponding to the prediction that will be made
            context_pos = context_pos[-1,:].unsqueeze(0).detach().numpy()
            #get fMRI final frame from input sequence (which will transform into the prediction)
            final = context[-1,:].unsqueeze(0).detach().numpy()
            pred_frame = predicted_fMRI[-1].unsqueeze(0).detach().numpy()
            #get output from the FFWD layer of the model (without the residual connection so that this is just the transform of the embedding to be added to the embeddding)
            fact = self.activations[-1]
            #concatenate the embedding transformations, and add the context_positional embeddings to the activations for transformations of the embedding (because this is a resnet)
            if add_first_and_final_frame:
                embedding_trajectory = np.concatenate((final,context_pos,self.activations+context_pos,pred_frame),axis=0)
            else:
                embedding_trajectory = np.concatenate((context_pos,self.activations+context_pos),axis=0)
            if i == 0:
                all_embedding_trajectories = embedding_trajectory
            else:
                all_embedding_trajectories = np.concatenate((all_embedding_trajectories,embedding_trajectory),axis=0)
            
            all_facts.append(fact)
            self.activations = []
            self.heads_just_done = False
            self.head_outputs_all = []
            self.head_num = 0
            start += self.block_size #non-overlapping blocks
            stop += self.block_size
            
        
        return all_embedding_trajectories, all_facts
    
def get_stack_labels(index_num, num_pred_steps, with_first_last):
    #get boolean indices for different layers of the embedding transformation
    #0 - positional encoding, 1 - layer norm(1), 2 - self attention, 3 - projection, 4 - layer norm(2), 5 - feed forward
    if with_first_last:
        labels = []
        for i in range(num_pred_steps):
            index = np.zeros(8)
            index[index_num+1] = 1
            #concatenate index to the end of labels
            labels = np.concatenate((labels,index),axis=0)
    else:
        labels = []
        for i in range(num_pred_steps):
            index = np.zeros(6)
            index[index_num] = 1
            #concatenate index to the end of labels
            labels = np.concatenate((labels,index),axis=0)
            
    #turn labels into boolean array
    labels = np.array(labels).astype(bool)
    return labels


def get_within_vs_between_embeddings(embedding_trajectories,num_subjects,num_pred_steps,with_first_last):
    
    #see if subject specific information exists in data
    r_between = np.full((num_subjects,num_subjects), np.nan)
    r_within = np.full((num_subjects), np.nan)
    num_scans = 4
    tempor = np.full((num_scans,num_scans), np.nan)
    if with_first_last: 
        embedding = embedding_trajectories.reshape(num_subjects,num_scans*num_pred_steps*8,200)
    else:
        embedding = embedding_trajectories.reshape(num_subjects,num_scans*num_pred_steps*6,200)
    indices = get_stack_labels(5, num_pred_steps, with_first_last)
    

    for subject_i in range(num_subjects):
        #calculate within subject similarity of embedding space
        for scan_i in range(num_scans):
            for scan_j in range(num_scans):
                if scan_i != scan_j:
                    #mean value of embedding
                    embedding_i = np.mean(embedding_trajectories[subject_i, scan_i,indices,:].squeeze().squeeze(),axis=0)
                    embedding_j = np.mean(embedding_trajectories[subject_i, scan_j,indices,:].squeeze().squeeze(),axis=0)
                    tempor[scan_i,scan_j] = np.corrcoef(embedding_i, embedding_j)[0,1]
        r_within[subject_i] = np.nanmean(tempor)
        
        for subject_j in range(num_subjects):
            if subject_i != subject_j:
                #reshape embedding trajectorys to (subject,scan*time,brain_regions)
                #mean value of embedding
                #embedding_i = np.mean(embedding[subject_i, :,:].squeeze(),axis=0)
                #embedding_j = np.mean(embedding[subject_j, :,:].squeeze(),axis=0)
                embedding_i = np.mean(embedding_trajectories[subject_i, 0,indices,:].squeeze().squeeze(),axis=0)
                embedding_j = np.mean(embedding_trajectories[subject_j, 0,indices,:].squeeze().squeeze(),axis=0)
                #correlate embedding i and embedding j 
                r_between[subject_i,subject_j]= np.corrcoef(embedding_i, embedding_j)[0,1]

    
    r_between_flat = r_between[~np.isnan(r_between)]
    r_within = r_within[~np.isnan(r_within)]

    #get mean difference
    mean_within = np.mean(r_within)
    mean_between = np.mean(r_between_flat)
    diff = mean_within - mean_between

    return r_within, r_between_flat, diff

def get_within_vs_between_facts(all_facts,num_subjects):
    
    #see if subject specific information exists in data
    r_between = np.full((num_subjects,num_subjects), np.nan)
    r_within = np.full((num_subjects), np.nan)
    num_scans = 4
    tempor = np.full((num_scans,num_scans), np.nan)
    for subject_i in range(num_subjects):
        #calculate within subject similarity of embedding space
        for scan_i in range(num_scans):
            for scan_j in range(num_scans):
                if scan_i != scan_j:
                    #mean value of embedding
                    embedding_i = np.mean(all_facts[subject_i, scan_i,:,:].squeeze().squeeze(),axis=0)
                    embedding_j = np.mean(all_facts[subject_i, scan_j,:,:].squeeze().squeeze(),axis=0)
                    tempor[scan_i,scan_j] = np.corrcoef(embedding_i, embedding_j)[0,1]
        r_within[subject_i] = np.nanmean(tempor)
        
        for subject_j in range(num_subjects):
            if subject_i != subject_j:
                #reshape embedding trajectorys to (subject,scan*time,brain_regions)
                #mean value of embedding
                #embedding_i = np.mean(embedding[subject_i, :,:].squeeze(),axis=0)
                #embedding_j = np.mean(embedding[subject_j, :,:].squeeze(),axis=0)
                embedding_i = np.mean(all_facts[subject_i, 0,:,:].squeeze().squeeze(),axis=0)
                embedding_j = np.mean(all_facts[subject_j, 0,:,:].squeeze().squeeze(),axis=0)
                #correlate embedding i and embedding j 
                r_between[subject_i,subject_j]= np.corrcoef(embedding_i, embedding_j)[0,1]

    
    r_between_flat = r_between[~np.isnan(r_between)]
    r_within = r_within[~np.isnan(r_within)]

    #get mean difference
    mean_within = np.mean(r_within)
    mean_between = np.mean(r_between_flat)
    diff = mean_within - mean_between

    return r_within, r_between_flat, diff

