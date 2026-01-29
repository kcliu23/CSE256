# CSE 256 PA1

**Author:** Kai-Cheng Liu  
**PID:** A69042222  
**Date:** January 2026

## Overview
This repository implements a Deep Averaging Network (DAN) for sentiment classification on the Stanford Sentiment Treebank. It systematically compares the performance of pre-trained GloVe embeddings versus randomly initialized embeddings trained from scratch. Additionally, it implements a Byte Pair Encoding (BPE) tokenizer to evaluate the efficacy of subword modeling on this task.

---

## 1. Requirements & Setup

### Environment Setup (Conda)
The project requires Python 3.6+ and PyTorch. The recommended setup using Conda is as follows:

```bash
# 1. Create a new environment
conda create -n cse256 python=3.8

# 2. Activate the environment
conda activate cse256

# 3. Install PyTorch (CPU version is sufficient)
conda install pytorch -c pytorch

# 4. Install other dependencies (Plotting & vectorization)
conda install matplotlib scikit-learn

```

### Directory Structure

Please ensure your directory is structured as follows. The `data/` folder must contain the provided dataset and GloVe embedding files.

```text
.
├── main.py                     # Entry point for training/evaluating all models
├── DANmodels.py                # Implementations of DAN and SubwordDAN
├── BOWmodels.py                # Baseline Bag-of-Words models
├── bpe.py                      # Byte Pair Encoding (BPE) implementation
├── sentiment_data.py           # Data loading utilities
├── utils.py                    # Indexer utilities
├── data/                       # Dataset directory
│   ├── train.txt
│   ├── dev.txt
│   ├── glove.6B.300d-relativized.txt
│   └── glove.6B.50d-relativized.txt
└── README.md

```

---

## 2. How to Run the Code

The `main.py` script handles all experiments. Use the `--model` flag to specify which part of the assignment to execute.

### Part 1: Deep Averaging Network (DAN)

This command runs the primary comparison between Transfer Learning and Random Initialization.

1. **GloVe Init:** Loads 300d pre-trained vectors.
2. **Random Init:** Initializes 300d vectors from scratch.
3. **Comparision:** Generates overlay plots for Accuracy and Loss.

```bash
python main.py --model DAN

```

* **Output:** Console logs for every epoch and plots saved to `results/`.
* **Approximate Runtime:** 2-4 minutes.

### Part 2: Byte Pair Encoding (BPE)

This command runs the subword modeling experiments.

1. **Tokenizer Training:** Trains BPE tokenizers on `train.txt` for vocab sizes 1k, 2k, and 5k.
2. **Model Training:** Trains a subword-DAN for each vocabulary size to find the optimal trade-off.

```bash
python main.py --model SUBWORDDAN

```

* **Output:** BPE merge statistics and training logs. Plots saved to `results/`.

### Baseline: Bag of Words (BOW)

To run the feed-forward baseline (comparing 2-layer vs 3-layer networks):

```bash
python main.py --model BOW

```

---

## 3. Output Files

All visualization results are automatically saved to the `results/` directory.

| File Name | Description |
| --- | --- |
| **`dan_all_accuracy.png`** | Validation Accuracy comparison: GloVe vs. Random. |
| **`dan_loss_convergence.png`** | Training Loss comparison: GloVe vs. Random. |
| **`bpe_results.png`** | Validation Accuracy for BPE vocab sizes (1k, 2k, 5k). |
| **`bpe_loss.png`** | Training Loss convergence for BPE vocab sizes. |
| **`dan_glove_full.png`** | Detailed Train/Dev curves for the GloVe model. |
| **`dan_random_full.png`** | Detailed Train/Dev curves for the Random model. |

---

## 4. Implementation Highlights

* **Strict Masked Averaging:** The DAN `forward()` method implements strict masking to ensure padding tokens (index 0) are excluded from both the sum and the count during averaging.
* **Input Dropout:** A dropout layer () is applied immediately after the averaging operation. This was critical for stabilizing the Random Initialization model.
* **Hyperparameters:** To ensure a fair comparison, both GloVe and Random models use identical settings:
* Hidden Size: 300
* Dropout: 0.3
* Learning Rate: 
* Weight Decay: 


* **Early Stopping:** Training monitors Dev Accuracy and stops if no improvement is seen for 15 epochs, restoring the best model weights.

