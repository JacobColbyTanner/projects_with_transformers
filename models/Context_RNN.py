# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
from utils import estimate_loss, get_batch

'''
# data loading
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
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
            # Import get_batch from main file
            from project5_main import get_batch
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
'''


def loss_to_bpc(loss):
    """Convert cross-entropy loss to bits per character"""
    return loss / math.log(2)


class ContextCTRNN(nn.Module):
    def __init__(self, context_size, embedding_size, action_embedding_size, vocab_size):
        super().__init__()
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.action_embedding_size = action_embedding_size

        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.embedding2context = nn.Embedding(
            vocab_size, self.embedding_size)
        self.embedding2actionable = nn.Embedding(
            vocab_size, self.embedding_size)
        self.context2context_map = nn.Linear(
            self.context_size, self.context_size * self.embedding_size)
        self.context2action_map = nn.Linear(
            self.context_size, self.embedding_size * self.action_embedding_size)
        self.output_layer = nn.Linear(
            self.action_embedding_size, vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):

        context_embedding = self.embedding2context(input)
        actionable_embedding = self.embedding2actionable(input)

        context_map = self.context2context_map(
            context).view(-1, self.context_size, self.embedding_size)

        transformed_context_embedding = torch.bmm(
            context_map, context_embedding.unsqueeze(-1)).squeeze(-1)

        # normalize transformed_context_embedding layer norm
        transformed_context_embedding = transformed_context_embedding / \
            torch.norm(transformed_context_embedding, p=2, dim=1).unsqueeze(1)

        # Compute action map from context, before the context update
        action_map = self.context2action_map(
            context).view(-1, self.action_embedding_size, self.embedding_size)

        context = (1-self.alpha) * context + self.alpha * \
            transformed_context_embedding

        # Ensure actionable_portion has the correct shape for batch matrix multiplication
        actionable_embedding = actionable_embedding.view(
            -1, self.context_size, 1)

        # Apply action map to input
        action_map_output = torch.bmm(
            action_map, actionable_embedding).squeeze(-1)

        output = self.output_layer(action_map_output)

        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1])
            context = context.to(input.device)
        else:
            context = context

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(
                    input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, context


class Context_Layers(nn.Module):
    def __init__(self, context_size, vocab_size, num_layers=2, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            ContextCTRNN(context_size, vocab_size, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x, targets, num_steps=1):
        current_input = x

        for layer in self.layers:
            logits, _ = layer(current_input, num_steps=num_steps)
            # Convert logits to token indices for next layer input
            if layer != self.layers[-1]:  # Don't convert if it's the last layer
                current_input = torch.argmax(logits, dim=2).long()

        # Use the final layer's output for loss calculation
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss


class ContextRNNNet(nn.Module):
    def __init__(self, context_size, embedding_size, action_embedding_size, vocab_size, **kwargs):
        super().__init__()
        self.rnn = ContextCTRNN(
            context_size, embedding_size, action_embedding_size, vocab_size, **kwargs)

    def forward(self, x, targets, num_steps=1):
        # Standard next-token prediction loss
        logits, _ = self.rnn(x)
        loss = F.cross_entropy(logits.permute(1, 2, 0), targets)

        return logits, loss

    def generate(self, x, max_new_tokens=2000, temperature=0.8, top_k=None, top_p=None):
        """Generate new tokens given a context."""
        context = None
        # Store all outputs
        outputs = []
        # Add initial input to outputs
        if x.dim() == 2:
            outputs.append(x[0])  # If x is [batch, seq], take first batch
        else:
            outputs.append(x)  # If x is just [seq]

        # Move to same device as model
        x = x.to(next(self.parameters()).device)

        for i in range(max_new_tokens):
            # Get logits for next token
            # Only use last token as input
            logits, context = self.rnn(x[:, -1:], context)
            logits = logits[-1].squeeze() / temperature  # Get final timestep

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            if top_k is not None:
                indices_to_remove = torch.topk(probs, k=top_k)[1]
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask[indices_to_remove] = True
                probs[~mask] = 0.0
                probs = probs / probs.sum()

            # Nucleus (top-p) sampling
            elif top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum()

            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, 1)
            outputs.append(next_token)
            x = next_token.unsqueeze(0)  # Add batch dimension back

        # Ensure all tensors have same dimensions before concatenating
        outputs = [o.view(1, -1) if o.dim() == 1 else o for o in outputs]
        return torch.cat(outputs, dim=1)
