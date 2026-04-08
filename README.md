# regular-irregular-sequence-classifier

**1D CNN for classifying binary pulse sequences as regular or irregular under noise, drift, and variable-length conditions.**

This repository contains a compact **1D convolutional neural network (CNN)** for detecting whether a binary pulse sequence is **regular** or **irregular**. The model is designed to work directly on raw sequences without handcrafted preprocessing and remains robust under realistic distortions such as noise, slow frequency drift, variable sequence length, and intermittent irregularity.

---

## Problem Overview

Each input is a binary sequence where:

- `0` = no pulse
- `1` = pulse

Examples:

- **Regular:** `000100001000010000100`
- **Irregular:** `0001001000000101000000100`

The task is to classify sequences according to whether the spacing between pulses follows an underlying regular pattern.

### Main challenges

The problem is harder than simple periodicity detection because the data may contain:

- slight frequency drift over time
- non-integer effective spacing
- random bit flips (`0 <-> 1`)
- variable-length sequences
- partially irregular segments within otherwise regular signals

---

## Approach

I use a **1D CNN** that learns pulse-spacing patterns directly from the raw binary sequence.

### Main components

- **Conv1D layers** to capture local and mid-range temporal structure
- **Different kernel sizes** to observe both short- and long-range spacing patterns
- **BatchNorm1d** for more stable and faster training
- **ReLU activations** for non-linearity
- **MaxPool1d** to reduce sequence length and enlarge the effective receptive field
- **AdaptiveAvgPool1d** to support variable-length inputs
- **Dropout in the classifier head** to reduce overfitting
- **Fully connected head** for final binary classification

The architecture was optimized with emphasis on:

- performance on the **hard** task
- parameter efficiency
- keeping the solution generic and task-compliant

---

## Final Model

The final selected model is **v5**:

- **Channels:** `1→8→16→24→24→32`
- **Head:** `32→16→1`
- **Parameters:** `24,145`
- **Easy AUC:** `0.9998`
- **Hard AUC:** `0.9542`

This model gave the best overall trade-off between accuracy and compactness.

---

## Training Configuration

The main training setup was:

- **Epochs:** 50
- **Early stopping:** patience = 5
- **Training set size:** 30,000
- **Validation set size:** 6,000
- **Batch size:** 64
- **Optimizer:** AdamW
- **Loss:** BCEWithLogitsLoss
- **Seed:** 42

Configuration file:

- `configs/train.yaml`

---

## Results

| Model | Channels | Head | # Params | Training Length | Best Val AUC | Easy AUC | Easy Acc | Hard AUC | Hard Acc | Efficiency (Hard AUC / 10k params) | Notes |
|------|----------|------|---------:|----------------:|-------------:|---------:|---------:|---------:|---------:|-----------------------------------:|-------|
| v1 | 1→16→32→64→64→96 | 96→32→1 | 168,417 | 17 ep | 0.9999 | 0.9999 | 0.9968 | 0.9662 | 0.9547 | 0.574 | Baseline |
| v2 | 1→8→16→32→32→64 | 64→32→1 | 51,393 | 25 ep | 1.0000 | 0.9999 | 0.9959 | 0.9625 | 0.9551 | 0.187 | Compression of layer dimensions |
| v3 | 1→8→16→32→32→64 | 64→1 | 49,345 | 34 ep | 0.9982 | 0.9979 | 0.9818 | 0.9414 | 0.9400 | 0.191 | Flattened head, too much compression |
| v4 | 1→8→16→32→32→64 | 64→16→1 | 50,337 | 12 ep | 0.9997 | 0.9994 | 0.9876 | 0.9473 | 0.9512 | 0.188 | Reduced head |
| v5 | 1→8→16→24→24→32 | 32→16→1 | 24,145 | 34 ep | 0.9999 | 0.9998 | 0.9957 | 0.9542 | 0.9503 | 0.395 | Best trade-off |
| v6 | 1→8→16→24→32 | 32→16→1 | 17,737 | 35 ep | 0.9985 | 0.9981 | 0.9836 | 0.9347 | 0.9418 | 0.523 | Too shallow, performance drop |

---

## Repository Structure

```text
.
├── Assignment.md
├── Description.md
├── Performance.md
├── Requirements.txt
├── configs
│   └── train.yaml
├── data
│   ├── eval_easy.json
│   └── eval_hard.json
├── models
│   ├── ....
└── src
    ├── evaluate.py
    ├── generator.py
    ├── model.py
    └── train.py
```

### File roles

- `Assignment.md` — original homework/task description
- `Description.md` — short methodological summary
- `Performance.md` — comparison of tested architectures
- `Requirements.txt` — Python dependencies
- `configs/train.yaml` — training configuration
- `data/` — evaluation datasets
- `models/` — archived model variants and checkpoints
- `src/model.py` — final model definition
- `src/train.py` — training script
- `src/evaluate.py` — evaluation script
- `src/generator.py` — synthetic data generation utilities

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/regular-irregular-sequence-classifier.git
cd regular-irregular-sequence-classifier
pip install -r Requirements.txt
```

If needed, you can also install packages manually:

```bash
pip install torch numpy matplotlib scikit-learn tqdm pyyaml
```

---

## Training

Run training from the project root:

```bash
python train.py --config ../configs/train.yaml
```

Depending on your script configuration, the best checkpoint will be saved during training.

---

## Evaluation

Evaluate on the easy dataset:

```bash
python src/evaluate.py --data data/eval_easy.json
```

Evaluate on the hard dataset:

```bash
python src/evaluate.py --data data/eval_hard.json
```

---

## Design Notes

This solution intentionally avoids task-specific handcrafted preprocessing.

### Not used

- FFT-based features
- manually extracted gap sequences
- hardcoded rules for pulse spacing

### Used

- raw binary sequence input
- standard neural network layers only
- generic convolutional feature extraction

This keeps the approach compliant with the assignment constraints and makes the solution more general.

---

## Summary

This project explores compact **1D CNN** architectures for detecting regularity in noisy pulse sequences. Multiple variants were tested to balance **hard-task AUC**, **parameter count**, and **generality**, with **v5** selected as the best overall model.
