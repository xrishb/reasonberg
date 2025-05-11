# Reasonberg

This repository contains implementations of different neural network architectures designed for multi-step reasoning and semantic understanding tasks. The implementations are built with PyTorch and focus on different approaches to knowledge integration and reasoning.

## Architectures

### 1. Clouds of Thoughts (original idea for this proejct)

A neural architecture that implements multi-step reasoning through attention-based knowledge matching and iterative refinement. It is supposed to be the architecture that i will continue my research on and refine it accordingly.

**Key Features:**
- Input understanding using linear transformation
- Knowledge matching using multi-head attention
- Multi-step reasoning through refinement layers
- Global attention mechanism for knowledge integration

**Use Cases:**
- Complex reasoning tasks requiring iterative thinking
- Knowledge-intensive NLP applications
- Multi-hop question answering

### 2. Semantic Tensor Reasoning (STRA) 

An architecture that implements semantic tensor reasoning by iteratively refining word vectors through semantic matching with a knowledge base.

**Key Features:**
- Word-by-word semantic matching with knowledge concepts
- Cosine similarity for concept matching
- Modular update layers for vector refinement
- Explicit vector fusion techniques

**Use Cases:**
- Semantic reasoning tasks
- Multi-hop inference
- Sentence classification and semantic entailment

## Requirements

Each architecture has its own requirements file, but the common dependencies are:

```
torch>=1.9.0
numpy>=1.20.0
```

Additional dependencies for specific architectures are found in their respective folders.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/xrishb/reasonberg.git
cd reasonberg
```

2. Install the required dependencies:
```bash
pip install -r archs/clouds_of_thoughts/requirements.txt
pip install -r archs/tensor_semantics/requirements.txt
```

## Usage

### Clouds of Thoughts

```python
from archs.clouds_of_thoughts.model import CloudsOfThoughts

# Initialize model
model = CloudsOfThoughts(
    embedding_dim=64,
    hidden_dim=128,
    knowledge_dim=100,
    reasoning_steps=3
)

# Process input
import torch
input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
output = model(input_tensor)
```

### Semantic Tensor Reasoning

```python
from archs.tensor_semantics.main import SemanticReasoner

# Initialize model
model = SemanticReasoner(
    embedding_dim=64,
    hidden_dim=128,
    num_concepts=100,
    reasoning_steps=4
)

# Process input
import torch
input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
output = model(input_tensor)
```

## Architecture Comparison

| Feature | Clouds of Thoughts | Tensor Semantics |
|---------|-------------------|------------------|
| Knowledge Integration | Global attention mechanism | Word-by-word matching |
| Matching Mechanism | Multi-head attention | Cosine similarity |
| Reasoning Process | Whole-tensor refinement | Individual word updates |
| Technical Approach | Transformer-style attention | Explicit similarity and fusion |

Both architectures are designed for multi-step reasoning but differ in how they match and update representations during the reasoning process.

## Future Work

- Integration with pre-trained embeddings (GloVe, BERT, etc.)
- Adding task-specific output layers for classification, QA, etc.
- Expanding the knowledge bases with external knowledge sources
- Creating hybrid architectures combining the strengths of both approaches
- Hybrid? or improving existing architectures.

## Acknowledgements

This project draws inspiration from various research papers in the fields of:
- Multi-hop reasoning in neural networks
- Knowledge integration in language models
- Semantic reasoning for NLP tasks 