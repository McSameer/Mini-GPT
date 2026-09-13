# Multi-Task NLP Transformer Implementation

A from-scratch implementation of a **GPT-2-style 124M-parameter Transformer** in PyTorch, featuring multi-head self-attention, causal masking, BPE tokenization, pretrained weight loading, text generation, downstream task fine-tuning, and automated LLM-based evaluation.

## Overview

This project implements a GPT-2-style decoder-only Transformer and extends it beyond language modeling to support **multiple NLP tasks**.

The implementation covers the complete NLP workflow:

**Tokenization → Transformer Pretraining/Weight Loading → Text Generation → Task Fine-Tuning → Automated Evaluation**

The model can generate text using configurable decoding strategies and can be adapted to downstream tasks such as **SMS spam classification** and **instruction following**.

## Key Features

* **124M-parameter GPT-2-style Transformer**

  * Decoder-only architecture
  * Multi-head causal self-attention
  * Position embeddings
  * Feed-forward Transformer blocks
  * Layer normalization and residual connections
  * Configurable model architecture

* **BPE Tokenization**

  * Byte Pair Encoding-based text tokenization
  * Token-to-ID and ID-to-token conversion
  * Text encoding and decoding pipeline

* **Causal Language Modeling**

  * Autoregressive next-token prediction
  * Causal attention masking to prevent access to future tokens
  * Cross-entropy language-modeling objective

* **Pretrained Weight Loading**

  * Compatible GPT-2-style parameter structure
  * Support for loading pretrained model weights
  * Enables inference and downstream fine-tuning without training the model from scratch

* **Text Generation**

  * Autoregressive token generation
  * Temperature scaling
  * Top-k sampling
  * Configurable generation length and sampling parameters

* **Multi-Task Fine-Tuning**

  * Adaptation of the pretrained Transformer to downstream NLP tasks
  * SMS spam classification
  * Instruction-following experiments using Alpaca-style data

* **Automated LLM Evaluation**

  * LLaMA-3 used as an external evaluator through Ollama
  * Deterministic evaluation prompts
  * Automated scoring of generated responses on a **0–100 scale**
  * Enables consistent comparison of instruction-following outputs

## Architecture

The core model follows the GPT-2 decoder-only Transformer architecture:

```text
                    Input Text
                        │
                        ▼
                  BPE Tokenizer
                        │
                        ▼
              Token IDs + Positions
                        │
                        ▼
        ┌───────────────────────────────┐
        │       Transformer Block       │
        │                               │
        │   Layer Normalization         │
        │           │                   │
        │           ▼                   │
        │   Multi-Head Self-Attention   │
        │      + Causal Mask            │
        │           │                   │
        │        Residual               │
        │           │                   │
        │           ▼                   │
        │   Layer Normalization         │
        │           │                   │
        │           ▼                   │
        │   Feed-Forward Network        │
        │           │                   │
        │        Residual               │
        └───────────────┬───────────────┘
                        │
                  Repeated Blocks
                        │
                        ▼
                 Language Model Head
                        │
                        ▼
                    Logits
                        │
                        ▼
               Sampling / Prediction
```

For downstream tasks, the Transformer representations are adapted to task-specific prediction heads.

## Model Configuration

The implementation targets a GPT-2-style **124M parameter** configuration.

| Component          | Configuration                    |
| ------------------ | -------------------------------- |
| Architecture       | Decoder-only Transformer         |
| Parameters         | ~124M                            |
| Attention          | Multi-head self-attention        |
| Attention Mask     | Causal                           |
| Framework          | PyTorch                          |
| Tokenization       | BPE                              |
| Objective          | Autoregressive language modeling |
| Generation         | Top-k + temperature sampling     |
| Pretrained Weights | Supported                        |

## Training & Fine-Tuning

The project supports an end-to-end training pipeline for adapting the Transformer to downstream NLP tasks.

### SMS Spam Classification

The pretrained Transformer was fine-tuned for binary SMS spam classification.

**Result:**

> **95.67% classification accuracy**

The task demonstrates how a generative Transformer architecture can be adapted from language modeling to supervised text classification.

### Alpaca Instruction Following

The model was also adapted using **Alpaca-style instruction data** to investigate its ability to follow natural-language instructions.

The instruction-following setup uses structured examples containing:

```text
Instruction
    ↓
Optional Input
    ↓
Expected Response
```

This allows the pretrained language model to be optimized for generating task-specific responses rather than only performing next-token prediction on generic text.

## Text Generation

The model supports autoregressive generation with configurable decoding parameters.

### Temperature Scaling

Temperature controls the sharpness of the probability distribution used during sampling.

* Lower temperature → more deterministic outputs
* Higher temperature → more diverse outputs

Example:

```python
generated_text = generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.7
)
```

### Top-k Sampling

Top-k sampling restricts the candidate tokens at each generation step to the `k` most probable tokens.

```python
generated_text = generate(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)
```

Combining temperature scaling with top-k sampling provides control over the trade-off between **generation quality, diversity, and determinism**.

## Automated LLM Evaluation

Instruction-following outputs are evaluated automatically using **LLaMA-3 through Ollama**.

Instead of relying exclusively on manual inspection, the project uses a standardized evaluator prompt to score generated responses.

```text
                 Model Response
                       │
                       ▼
                Evaluation Prompt
                       │
                       ▼
              LLaMA-3 via Ollama
                       │
                       ▼
                 Score: 0–100
```

The evaluation pipeline is designed to make model comparisons more consistent by keeping the evaluation criteria and prompting procedure fixed.

### Evaluation Characteristics

* LLaMA-3-based evaluator
* Local inference through Ollama
* Standardized evaluation prompt
* Automated response scoring
* Numerical score ranging from **0 to 100**
* Suitable for comparing different model checkpoints or fine-tuning configurations

## Project Structure

```text
.
├── model/
│   ├── transformer.py          # GPT-2-style Transformer implementation
│   ├── attention.py            # Multi-head self-attention
│   └── layers.py               # Transformer components
│
├── tokenizer/
│   └── tokenizer.py            # BPE encoding/decoding
│
├── training/
│   ├── train.py                # Language-model training
│   └── finetune.py             # Downstream task fine-tuning
│
├── generation/
│   └── generate.py              # Autoregressive text generation
│
├── evaluation/
│   └── evaluate.py              # LLaMA-3/Ollama evaluation
│
├── data/
│   └── README.md                # Dataset information
│
├── notebooks/
│   └── experiments.ipynb        # Experiments and analysis
│
├── requirements.txt
└── README.md
```

> The exact directory structure may differ depending on the implementation.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project is implemented primarily using:

* Python
* PyTorch
* NumPy
* Pandas
* BPE tokenizer implementation
* Ollama
* LLaMA-3

A GPU with sufficient VRAM is recommended for training and fine-tuning the 124M-parameter model.

## Usage

### 1. Load the Model

```python
from model import GPTModel

model = GPTModel(config)
```

### 2. Load Pretrained Weights

```python
model.load_state_dict(checkpoint)
model.eval()
```

### 3. Generate Text

```python
text = generate(
    model,
    prompt="Artificial intelligence is",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)

print(text)
```

### 4. Fine-Tune on a Downstream Task

```bash
python training/finetune.py \
    --task sms_spam \
    --epochs 5
```

### 5. Evaluate Instruction Following

Start Ollama and make the LLaMA-3 model available locally:

```bash
ollama run llama3
```

Then run:

```bash
python evaluation/evaluate.py
```

The evaluator processes generated responses and produces a numerical score for each example.

## Results

### SMS Spam Classification

| Task                    |   Metric |     Result |
| ----------------------- | -------: | ---------: |
| SMS Spam Classification | Accuracy | **95.67%** |

### Instruction Following

Instruction-following performance is evaluated using an automated LLaMA-3 evaluator that assigns each generated response a score between **0 and 100**.

This provides a consistent quantitative framework for comparing model outputs across different experiments and fine-tuning configurations.

## Technical Highlights

### Multi-Head Self-Attention

The model divides the hidden representation into multiple attention heads, allowing different heads to learn different relationships between tokens.

For each head:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
    = softmax(QKᵀ / √dₖ) V
```

A causal mask is applied before the softmax operation to ensure that each token can only attend to previous tokens and itself.

### Causal Masking

For autoregressive generation, future tokens must remain inaccessible during training.

The attention matrix therefore follows a lower-triangular structure:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This enables parallel training while preserving the autoregressive constraint used during generation.

### Transfer Learning

Rather than training every downstream model from scratch, the pretrained Transformer provides reusable language representations that can be adapted to specialized NLP tasks.

This enables the same underlying architecture to support:

```text
              GPT-2 Transformer
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Classification       Instruction
       Head                 Tuning
          │                     │
          ▼                     ▼
     SMS Spam             Alpaca-style
    Classification          Responses
```

## Evaluation Methodology

For instruction-following experiments, generated responses are passed to LLaMA-3 along with the corresponding instruction and evaluation criteria.

The evaluator produces a numerical score:

```text
0   ─────────────────────────── 100
Poor                         Excellent
```

Using a fixed evaluation procedure reduces subjective variation during manual assessment and makes it easier to compare multiple checkpoints or generation configurations.


The architecture is based on the Transformer/GPT family of decoder-only language models and is implemented using PyTorch for educational and experimental purposes.
