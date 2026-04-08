# Model Description

## Overview

I used a **1D convolutional neural network (CNN)** to classify binary sequences as **regular** or **irregular** based on the spacing between pulses (periodicity). The model works directly with the raw input sequence and does not use any handcrafted preprocessing, FFT, or manually extracted gap features. The main idea is that convolutional layers can learn different ranges of temporal patterns and pooling layers help the network build a more global view of the sequence. This is useful because regularity in the signal is determined by repeated spacing patterns, but the task also includes frequency drift, noise, variable sequence length, and partially irregular segments.

## Architecture

The model is based on several stacked **Conv1D** layers with **BatchNorm1d** and **ReLU** activations. I experimented with different channel sizes and head designs to balance performance and parameter count.

Main techniques used:

- **Conv1D layers** to detect pulse-spacing patterns directly from the sequence.
- **Different kernel sizes** to capture both short-range and longer-range structure.
- **BatchNorm1d** to stabilize training.
- **MaxPool1d** to gradually reduce sequence length and enlarge the effective receptive field.
- **AdaptiveAvgPool1d** to aggregate features and support variable-length sequences in the hard task.
- **Dropout** to reduce overfitting.
- **Fully connected output head** for final binary classificatio.

## Design Strategy

My goal was not only to maximize AUC, but also to find a good trade-off between **hard-task performance** and **number of parameters**. I therefore tested several model variants with different:

- channel widths
- network depth
- classification heads

A useful comparison metric was **efficiency = Hard AUC / (number of parameters / 10k)**. This helped me compare performance relative to model size.

## Training Configuration

- **Epochs:** 50
- **Early stopping:** patience = 5
- **Training set size:** 30k
- **Validation set size:** 6k
- **Batch size:** 64
- **Optimizer:** AdamW
- **Loss function:** BCEWithLogitsLoss
- **Random seed:** 42

## Performance of Various Models

| Model | Channels | Head | #Params | Training Length | Best Val AUC | Easy AUC | Easy Acc | Hard AUC | Hard Acc | Efficiency (Hard AUC / 10k params) | Notes |
|------|----------|------|---------:|----------------:|-------------:|---------:|---------:|---------:|---------:|-----------------------------------:|-------|
| v1 | 1→16→32→64→64→96 | 96→32→1 | 168417 | 17 ep | 0.9999 | 0.9999 | 0.9968 | 0.9662 | 0.9547 | 0.0574 | Baseline |
| v2 | 1→8→16→32→32→64 | 64→32→1 | 51393 | 25 ep | 1.0000 | 0.9999 | 0.9959 | 0.9625 | 0.9551 | 0.187 | Compression of layer dimensions |
| v3 | 1→8→16→32→32→64 | 64→1 | 49345 | 34 ep | 0.9982 | 0.9979 | 0.9818 | 0.9414 | 0.9400 | 0.191 | Flattened head, reduction of AUC, too much compression |
| v4 | 1→8→16→32→32→64 | 64→16→1 | 50337 | 12 ep | 0.9997 | 0.9994 | 0.9876 | 0.9473 | 0.9512 | 0.188 | Reduced head |
| v5 | 1→8→16→24→24→32 | 32→16→1 | 24145 | 34 ep | 0.9999 | 0.9998 | 0.9957 | 0.9542 | 0.9503 | 0.395 | Best trade-off |
| v6 | 1→8→16→24→32 | 32→16→1 | 17737 | 35 ep | 0.9985 | 0.9981 | 0.9836 | 0.9347 | 0.9418 | 0.523 | Too shallow, performance drop |

## Conclusion

The best overall trade-off was **model v5**. It achieved strong performance on both easy and hard evaluation sets while keeping the parameter count much lower than the baseline. Compared to larger models, it preserved most of the performance while being considerably more efficient. In summary, I used a compact **1D CNN** with pooling, normalization, adaptive averaging, and dropout to solve the task directly from raw binary sequences. The model was designed to remain generic, parameter-efficient, and robust to noise, frequency drift, variable sequence length, and partial irregularity. Codes and models can be found of [GitHub](https://github.com/David-9999/regular-irregular-sequence-classifier).
 
