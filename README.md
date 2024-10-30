#Note that the initial code used for the transformer model used here comes from a tutorial by Andrej Karpathy
#Original repo: https://github.com/karpathy/ng-video-lecture.git
#Original video: https://youtu.be/kCc8FmEb1nY?si=LRlFNDmZms70MkDe
(project0_main.py implements the example from this tutorial)

Here, I will be teaching myself how to use transformer models by training transformers to solve a number of sequence problems.

1st: I will be performing next word prediction/generation on Shakespeare writing (as in the tutorial above), but I will be changing the embedding from letter embeddings to word or partial word embeddings using different packages

2nd: I will predict the next time step of neural activity by training a transformer on fMRI data. This enable the transformer to create simulated time series data. Additionally, I investigate how the attention weights change across training. I find that initially, less training results in attention weights that focus primarily on the latest timestep to predict the next time step, but with more iterations of training the attention weights start to look periodic as though they are track where the peaks/troughs of fMRI oscillations occur. This appears to allow these models to generate simulated time series for a longer period of time before it starts exploding (when activity stops oscillating and instead continues to increase).

3rd: I will use the Maestro dataset to train a transformer to generate classical piano music from sequences of MIDI controller inputs to a piano

4th: Continuing the previous project on modeling classical piano music, I will use an altered version of self-attention, referred to as "relative self-attention" in order to improve model performance. 

This comes from the following paper by several co-authors of the original paper on transformers:
Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., ... & Eck, D. (2018). Music transformer. arXiv preprint arXiv:1809.04281.

In this paper they extend the relative positional embedding (paper below), to also train relative time and relative pitch embeddings. I suspect this is done in a manner very similar to the relative positional embeddings, but instead of operating over positional distances to generate a distance index matrix, it does so over time or pitch. Here is the original relative positional embedding paper:
Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. arXiv preprint arXiv:1803.02155.


Additionally, this paper uses a different data representation for MIDI involving 413 dimensional one hot encoding vectors representing different Note on, Note off, Velocity, and Time events. This comes from the following paper (RNN paper): 
Oore, S., Simon, I., Dieleman, S., Eck, D., & Simonyan, K. (2020). This time with feeling: Learning expressive musical performance. Neural Computing and Applications, 32, 955-967.




