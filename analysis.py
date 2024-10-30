import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch.nn.functional as F
import os

# Load the model and dataset


def load_model_and_data(model_path, data_path, sequence_length=1000):
    """Load saved model and a random portion of data"""
    # Load checkpoint
    checkpoint = torch.load(model_path)

    # Load character mappings
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    # Load full text and select random chunk
    with open(data_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Generate random start position
    max_start = len(full_text) - sequence_length
    start_idx = torch.randint(0, max_start, (1,)).item()
    text = full_text[start_idx:start_idx + sequence_length]

    # Encode text
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long).unsqueeze(0)

    print(f"Selected text from position {start_idx}:")
    print(f"Preview: {text[:100]}...")  # Show first 100 chars

    return checkpoint, data, chars, stoi, itos, text


def get_context_states(model, data):
    """Run model and collect context states"""
    model.eval()
    with torch.no_grad():
        states = []
        context = None

        # Process sequence and collect states
        for i in range(data.size(1)):
            x = data[:, i:i+1]
            logits, context = model.rnn(x, context)
            states.append(context.cpu().numpy())

    return np.array(states).squeeze()


def plot_pca_trajectory(states, text, save_path=None, filter_spaces=False):
    """Create PCA plot of context states, optionally filtering spaces/newlines"""
    if filter_spaces:
        # Create mask for non-space characters
        mask = [not (c.isspace() or c == '\n') for c in text]
        filtered_states = states[mask]
        filtered_text = ''.join(
            [c for c in text if not (c.isspace() or c == '\n')])
    else:
        filtered_states = states
        filtered_text = text

    # Compute PCA
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(filtered_states)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot trajectory
    plt.plot(states_2d[:, 0], states_2d[:, 1], 'b-', alpha=0.3)
    plt.scatter(states_2d[:, 0], states_2d[:, 1], c=range(len(states_2d)),
                cmap='viridis', alpha=0.5)

    # Add some text labels at regular intervals
    for i in range(0, len(filtered_text), len(filtered_text)//50):
        plt.annotate(filtered_text[i-9: i+1],
                     (states_2d[i, 0], states_2d[i, 1]))

    plt.title('PCA Trajectory of Context States' +
              (' (Filtered)' if filter_spaces else ''))
    plt.xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.3f})')

    if save_path:
        base, ext = os.path.splitext(save_path)
        if filter_spaces:
            save_path = f"{base}_filtered{ext}"
        plt.savefig(save_path)
    plt.show()


def plot_recurrence(states, text, window_size=1, threshold=None, save_path=None):
    """Create recurrence plot from context states with character labels"""
    # Compute distance matrix
    distances = torch.cdist(torch.tensor(states), torch.tensor(states))

    if threshold is None:
        # Use mean distance as threshold
        threshold = distances.mean()/2

    # Create recurrence matrix
    recurrence = distances < threshold

    # Plot
    plt.figure(figsize=(12, 12))

    # Plot the recurrence matrix
    plt.imshow(recurrence, cmap='binary')

    # Add character labels
    # Only show a subset of ticks if sequence is long
    step = 1  # max(len(text), 1)  # Show ~20 ticks
    tick_positions = range(0, len(text), step)
    tick_labels = [text[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=0)
    plt.yticks(tick_positions, tick_labels)

    plt.title('Recurrence Plot')
    plt.xlabel('Characters')
    plt.ylabel('Characters')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    tick_labels = [text[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(tick_positions, tick_labels)

    plt.title('Recurrence Plot')
    plt.xlabel('Characters')
    plt.ylabel('Characters')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # Paths
    model_path = 'saved_models/text8_Context_Heads_best.pt'
    data_path = 'text8'
    sequence_length = 150  # Length of sequence to analyze

    # Create output directory
    Path('analysis_outputs').mkdir(exist_ok=True)

    # Load model and data
    checkpoint, data, chars, stoi, itos, text = load_model_and_data(
        model_path, data_path, sequence_length)

    # Get context size from the embedding layer's weight shape
    context_size = checkpoint['model_state_dict']['rnn.embedding2context.weight'].shape[1]

    # Recreate model with extracted context_size
    from models.Context_RNN import ContextRNNNet
    model = ContextRNNNet(context_size, len(chars))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get context states
    states = get_context_states(model, data)

    # Create both visualizations
    plot_pca_trajectory(states, text,
                        save_path='analysis_outputs/pca_trajectory.png',
                        filter_spaces=False)
    plot_pca_trajectory(states, text,
                        save_path='analysis_outputs/pca_trajectory.png',
                        filter_spaces=True)

    # Create visualizations
    plot_recurrence(
        states, text, save_path='analysis_outputs/recurrence_plot.png')

    # Print some statistics
    print(f"Analyzed sequence length: {sequence_length}")
    print(f"Context state dimension: {states.shape[1]}")
    print(f"Number of unique characters: {len(chars)}")


if __name__ == "__main__":
    main()
