import torch
import math


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # Check if we're using a chunked dataset
    if hasattr(train_data, 'get_chunk'):
        data = train_data if split == 'train' else val_data

        # Get random positions for our batch
        chunk_size = block_size + 1  # +1 because we need both input and target
        positions = torch.randint(len(data) - chunk_size, (batch_size,))

        # Get chunks and prepare tensors
        x = torch.zeros((batch_size, block_size), dtype=torch.long)
        y = torch.zeros((batch_size, block_size), dtype=torch.long)

        for i, pos in enumerate(positions):
            chunk = data.get_chunk(pos + data.start, chunk_size)
            x[i] = chunk[:block_size]
            y[i] = chunk[1:block_size+1]

        return x.to(device), y.to(device)
    else:
        # Original code for smaller datasets
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device, model_select):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data,
                             block_size, batch_size, device)
            if model_select == 'Context_Heads':
                logits, loss = model(X.T, Y)
            else:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def loss_to_bpc(loss):
    """Convert cross-entropy loss to bits per character"""
    return loss / math.log(2)


def decode(tokens, itos):
    """Convert token indices back to characters"""
    return ''.join([itos[i] for i in tokens])
