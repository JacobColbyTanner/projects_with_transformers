import torch
import torch.nn as nn
from torch.nn import functional as F
from models.Context_RNN import Context_Layers, estimate_loss, loss_to_bpc, get_batch, ContextRNNNet
from models.LSTM_model import LSTMModel, VanillaRNN
import os
import requests
from pathlib import Path
import math
from utils import estimate_loss, get_batch, decode, loss_to_bpc
import random


# TODO:
# Proper arithmetic learning would require:
# 1. Batch generation that preserves complete problems
# 2. Loss calculation only on the answer portion
# 3. Possibly curriculum learning (easier problems first)
# 4. Proper evaluation metrics for numerical accuracy


# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 150
n_head = 1
# LSTM/RNN hyperparameters
hidden_size = 150
num_layers = 1
# ------------

torch.manual_seed(1337)


def download_dataset(dataset_name):
    if dataset_name == 'arithmetic':
        # Generate arithmetic dataset
        print("Generating arithmetic dataset...")
        text = generate_arithmetic_dataset()
        filename = 'arithmetic.txt'
        with open(filename, 'w') as f:
            f.write(text)
        return filename

    elif dataset_name == 'tinyshakespeare':
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        filename = 'input.txt'
    elif dataset_name == 'enwik8':
        url = 'https://data.deepai.org/enwik8.zip'
        filename = 'enwik8'
    elif dataset_name == 'text8':
        url = 'http://mattmahoney.net/dc/text8.zip'
        filename = 'text8'
    elif dataset_name == 'ptb':
        url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
        filename = 'ptb.txt'

    if not os.path.exists(filename):
        print(f"Downloading {dataset_name}...")
        response = requests.get(url)
        if filename.endswith('.zip') or url.endswith('.zip'):
            zip_filename = filename + \
                '.zip' if not filename.endswith('.zip') else filename
            with open(zip_filename, 'wb') as f:
                f.write(response.content)
            import zipfile
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall('.')
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)

    return filename


def generate_arithmetic_dataset(num_examples=1000000, max_digits=5, operations=['+', '-', '*']):
    """Generate arithmetic problems and solutions"""
    problems = []
    for _ in range(num_examples):
        a = random.randint(0, 10**max_digits)
        b = random.randint(0, 10**max_digits)
        op = random.choice(operations)

        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:
            result = a * b

        problem = f"{a}{op}{b}={result}\n"
        problems.append(problem)
    return ''.join(problems)


class ChunkedDataset:
    def __init__(self, filename, chunk_size=1000000, split='train'):
        self.filename = filename
        self.chunk_size = chunk_size
        self.total_size = os.path.getsize(filename)

        # Define splits (90% train, 5% val, 5% test)
        if split == 'train':
            self.start = 0
            self.end = int(0.9 * self.total_size)
        elif split == 'val':
            self.start = int(0.9 * self.total_size)
            self.end = int(0.95 * self.total_size)
        else:  # test
            self.start = int(0.95 * self.total_size)
            self.end = self.total_size

        # Build vocabulary from multiple chunks across the file
        self.chars = self._build_vocabulary()
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {self.chars}")

    def _build_vocabulary(self):
        """Build vocabulary by sampling from different parts of the file"""
        chars = set()
        sample_size = 1000000  # Read 1MB from each sample point
        num_samples = 10  # Number of points to sample from

        with open(self.filename, 'rb') as f:  # Open in binary mode
            # Sample from beginning
            chars.update(f.read(sample_size).decode('latin-1'))

            # Sample from different points in the file
            file_size = self.total_size
            for i in range(1, num_samples-1):
                position = (file_size * i) // num_samples
                f.seek(position)
                chars.update(f.read(sample_size).decode('latin-1'))

            # Sample from end
            f.seek(max(0, file_size - sample_size))
            chars.update(f.read(sample_size).decode('latin-1'))

        return sorted(list(chars))

    def __len__(self):
        return self.end - self.start

    def get_chunk(self, start_pos, size):
        with open(self.filename, 'rb') as f:  # Open in binary mode
            f.seek(start_pos)
            chunk = f.read(size).decode('latin-1')
            return torch.tensor([self.stoi[c] for c in chunk], dtype=torch.long)


# Choose dataset and model
# 'tinyshakespeare', 'enwik8', 'text8', 'ptb', or 'arithmetic'
dataset_name = 'ptb'
model_select = 'Context_Heads'  # 'LSTM', 'Context_Heads', or 'VanillaRNN'

# Create models directory if it doesn't exist
Path('saved_models').mkdir(exist_ok=True)

# Define model save path
model_save_path = f'saved_models/{dataset_name}_{model_select}_model.pt'

# Download/generate and load the selected dataset
filename = download_dataset(dataset_name)

# Load based on dataset size
if dataset_name in ['text8', 'enwik8']:
    # Use chunked loading for large datasets
    train_data = ChunkedDataset(filename, split='train')
    val_data = ChunkedDataset(filename, split='val')
    chars = train_data.chars
    vocab_size = train_data.vocab_size
    stoi = train_data.stoi
    itos = train_data.itos
else:
    # Original loading for smaller datasets
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]


if model_select == 'Context_Heads':
    # ContextRNNNet(context_size, embedding_size, action_embedding_size, vocab_size)
    model = ContextRNNNet(hidden_size, n_embd, vocab_size,
                          vocab_size)
elif model_select == 'LSTM':
    model = LSTMModel(vocab_size, n_embd, hidden_size, num_layers=num_layers)
elif model_select == 'VanillaRNN':
    model = VanillaRNN(vocab_size, n_embd, hidden_size, num_layers=num_layers)

model = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# Training configuration
training_config = {
    'learning_rate': 1e-3,  # Starting learning rate
    'min_lr': 1e-4,        # Minimum learning rate
    'grad_clip': 0.25,
    'weight_decay': 0.001,
    'max_epochs': 100,
    'patience': 10,
    'batch_size': 4,
    'block_size': 512,
    'eval_interval': 50
}


# Calculate iterations per epoch and total iterations


def calc_num_iters(data_size, batch_size, max_epochs):
    iters_per_epoch = data_size // batch_size
    total_iters = iters_per_epoch * max_epochs
    return total_iters, iters_per_epoch


# Calculate total iterations
total_iters, iters_per_epoch = calc_num_iters(
    len(train_data),
    training_config['batch_size'],
    training_config['max_epochs']
)


class CosineScheduler:
    def __init__(self, optimizer, min_lr, max_lr, total_iters):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters
        self.current_step = 0

        # Set initial learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = max_lr

    def step(self):
        self.current_step += 1
        # Cosine decay from max_lr to min_lr
        progress = self.current_step / self.total_iters
        lr = self.min_lr + 0.5 * \
            (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config['learning_rate'],
    weight_decay=training_config['weight_decay'],
    betas=(0.9, 0.999)
)

scheduler = CosineScheduler(
    optimizer,
    min_lr=training_config['min_lr'],
    max_lr=training_config['learning_rate'],
    total_iters=total_iters
)

# Training stats
best_val_loss = float('inf')
training_stats = {'train_loss': [], 'val_loss': [],
                  'train_bpc': [], 'val_bpc': [],
                  'lr': []}

# Training loop with epoch tracking
current_epoch = 0
for iter in range(total_iters):
    # Update epoch counter
    if iter % iters_per_epoch == 0:
        current_epoch += 1
        print(
            f"\nStarting epoch {current_epoch}/{training_config['max_epochs']}")

    # Get current learning rate
    current_lr = scheduler.step()

    # Evaluation
    if iter % training_config['eval_interval'] == 0 or iter == total_iters - 1:
        losses = estimate_loss(model, eval_iters, train_data, val_data,
                               block_size, batch_size, device, model_select)
        print(
            f"epoch {current_epoch}, step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(
            f"        train bpc {loss_to_bpc(losses['train']):.4f}, val bpc {loss_to_bpc(losses['val']):.4f}")
        print(f"        lr {current_lr:.6f}")

        # Save training stats
        training_stats['train_loss'].append(losses['train'])
        training_stats['val_loss'].append(losses['val'])
        training_stats['train_bpc'].append(loss_to_bpc(losses['train']))
        training_stats['val_bpc'].append(loss_to_bpc(losses['val']))
        training_stats['lr'].append(current_lr)

        # Save if best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': iter,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.__dict__,
                'val_loss': best_val_loss,
                'val_bpc': loss_to_bpc(best_val_loss),
                'training_stats': training_stats,
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
            }, model_save_path)
            print(f"New best model! Val BPC: {loss_to_bpc(best_val_loss):.3f}")

    # Training step
    xb, yb = get_batch('train', train_data, val_data,
                       block_size, batch_size, device)

    # evaluate the loss
    if model_select == 'Context_Heads':
        logits, loss = model(xb.T, yb)
        # print("loss: ", loss.item())
    else:
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), training_config['grad_clip'])

    # Update
    optimizer.step()

# Load the best model for generation
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

# generate from the model
if dataset_name != 'arithmetic':
    # Allow for custom initial context
    initial_text = "The quick brown fox jumps over the lazy dog"
    if initial_text.strip():
        # Preprocess text based on dataset
        if dataset_name == 'text8':
            # Convert to lowercase and remove any characters not in text8's vocabulary
            initial_text = ''.join(c.lower()
                                   for c in initial_text if c.lower() in stoi)
            # Show user the processed text
            print(f"Preprocessed text: {initial_text}")

        context = torch.tensor(
            [[stoi[c] for c in initial_text]], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(
        context, max_new_tokens=500, temperature=0.8)[0].tolist()
    if initial_text.strip():
        # Skip the initial context in output if it was provided
        generated = generated[len(initial_text):]
    print(decode(generated, itos))
else:
    print("\nTesting arithmetic abilities:")
    for _ in range(5):  # Test 5 examples
        # Generate random numbers and operation
        a = random.randint(0, 99999)
        b = random.randint(0, 99999)
        op = random.choice(['+', '-'])

        # Calculate actual result
        true_result = a + b if op == '+' else a - b

        # Calculate number of digits needed for answer
        num_digits = len(str(true_result))

        # Create input context
        input_text = f"{a}{op}{b}="
        context = torch.tensor(
            [[stoi[c] for c in input_text]], dtype=torch.long, device=device)

        # Let model generate exactly the number of digits needed
        generated = model.generate(
            context, max_new_tokens=num_digits, temperature=0.7)[0].tolist()
        # Skip the input context
        generated_text = decode(generated[len(input_text):], itos)

        # Print results
        print(f"Problem: {input_text}")
        print(f"Model output: {generated_text}")
        print(f"True answer: {true_result}")
        print(f"{'✓' if generated_text == str(true_result) else '✗'}\n")

# At the end of training, print final stats
print("\nTraining Complete!")
print(f"Model: {model_select}")
print(f"Dataset: {dataset_name}")
print(f"Best Validation BPC: {loss_to_bpc(best_val_loss):.3f}")
print(f"Final Train BPC: {training_stats['train_bpc'][-1]:.3f}")
print(f"Final Val BPC: {training_stats['val_bpc'][-1]:.3f}")
