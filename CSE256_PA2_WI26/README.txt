# CSE 256 PA2: Transformer Blocks

### Project Overview

This project involves the from-scratch implementation of Transformer components in PyTorch to perform two distinct NLP tasks:

1. 
**Speech Classification**: A Transformer Encoder trained jointly with a feedforward classifier to identify speakers (Obama, W. Bush, or G.H. Bush).


2. 
**Language Modeling**: A GPT-like Transformer Decoder pretrained on an autoregressive task to predict the next word in a sequence.


3. 
**Architectural Exploration**: Implementation and evaluation of advanced attention mechanisms, including AliBi Positional Encoding, Sparse Window Attention, and Disentangled Attention.



---

### Prerequisites

The project requires the following libraries:

* **Python 3.8+**
* 
**PyTorch**: For model building and tensor operations.


* 
**NLTK**: Used by the tokenizer for word-level splitting.


* 
**Matplotlib**: Used by `utilities.py` to visualize attention matrices.



---

### Installation

Ensure you have the necessary dependencies installed:

```bash
pip install torch nltk matplotlib

```

---

### How to Run

The implementation follows the requested interface, allowing each part to be executed via `main.py` with the `--part` argument.

#### **Part 1: Encoder & Speech Classification**

Trains the encoder and classifier for 15 epochs and reports accuracy on `test_CLS.txt`.

```bash
python main.py --part part1

```

#### **Part 2: Decoder & Language Modeling**

Pretrains the decoder for 500 iterations and evaluates perplexity on politician-specific test sets.

```bash
python main.py --part part2

```

#### **Part 3: Architectural Exploration**

Runs the experimental suite comparing model scaling, AliBi, Sparse, and Disentangled attention patterns.

```bash
python main.py --part part3

```

---

### Implementation Details

* 
**Scratch Implementation**: All Transformer blocks (Attention, FeedForward, Blocks, Encoder, Decoder) are implemented without using high-level library modules like `nn.Transformer`.


* 
**Positional Encoding**: The base models use absolute positional embeddings added to token embeddings.


* 
**Causal Masking**: The decoder uses a lower-triangular mask to prevent peeking at future tokens.


* **Evaluation**:
* 
**Classification**: Accuracy on a three-way task.


* 
**Language Modeling**: Perplexity as .





---

### File Descriptions

* 
`main.py`: Contains hyperparameters, training loops, and the entry point for all assignment parts.


* 
`transformer.py`: Contains the core logic for the Encoder, Decoder, Multi-Head Attention, and FeedForward blocks.


* 
`tokenizer.py`: A simple word-level tokenizer using NLTK.


* 
`dataset.py`: Defines PyTorch `Dataset` classes for both classification and language modeling tasks.


* 
`utilities.py`: Provides helper functions to verify attention matrix normalization and generate attention heatmaps.



---

### Default Hyperparameters

* **Batch Size**: 16
* **Block Size**: 32
* **Embedding Dimension**: 64 (Base)
* **Attention Heads**: 2
* **Layers**: 4
* **Learning Rate**: 1e-3

---