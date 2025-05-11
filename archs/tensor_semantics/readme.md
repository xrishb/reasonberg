# **Semantic Tensor Reasoning Arch (STRA)**

This version implements **semantic tensor reasoning** using **PyTorch**. The core idea is to store word embeddings in a tensor, apply iterative updates based on semantic matching with a knowledge base, and refine sentence representations over multiple reasoning steps.

## **Overview**

The system utilizes an architecture where sentences are represented as 3D tensors, with word vectors (embeddings) in each cell. The reasoning process works by iteratively updating these vectors, matching each word to a semantic concept from an external knowledge bank, and refining the representations over several steps. This model is suited for tasks that involve **semantic reasoning** and **multi-hop inference**.

## **Key Features**

* **Semantic Reasoning**: The model iterates over the sentence tensor, updating the word representations based on external semantic knowledge.
* **External Knowledge Bank**: A matrix of concept embeddings is used to match and refine the word vectors during reasoning.
* **Cosine Similarity Matching**: The system uses cosine similarity to find related concepts in the knowledge bank for each word and refines the word's vector accordingly.
* **Modular Update Layer**: A learnable update layer that combines the word vector and the matched concept for refining the tensor.

## **Components**

### 1. **`SemanticReasoner` Class**

This is the main class that implements the reasoning model. It takes a sentence tensor, applies semantic matching with the knowledge bank, and iteratively updates the tensor.

#### Methods

* **`__init__()`**: Initializes the model with parameters such as embedding dimension, hidden dimension, knowledge bank size, and reasoning steps.
* **`forward()`**: Performs forward propagation by iterating over the tensor and applying the reasoning protocol, refining the tensor step-by-step.

### 2. **Knowledge Bank**

The knowledge bank is a set of learnable vectors representing concepts that words in the sentence are matched against. This is used to refine the sentence tensor during each reasoning step.

### 3. **Update Layer**

A neural network layer used to fuse the original word vector with the matched concept from the knowledge bank, producing an updated word vector.

## **Installation**

To run this code, you need to install **PyTorch**. You can install it using:

```bash
pip install torch
```

## **Usage**

### Running the Model

1. Clone this repository and navigate to the project folder.
2. Run the script `semantic_tensor_reasoning.py` using Python:

```bash
python semantic_tensor_reasoning.py
```

### Example Output

After running the script, you should see the input tensor and output tensor shapes:

```bash
Input shape:  torch.Size([2, 6, 64])
Output shape: torch.Size([2, 6, 64])
```

The model processes the input sentence tensor (randomly initialized for this example) and outputs the refined tensor after a series of reasoning steps.

## **Parameters**

* **`embedding_dim`**: The dimension of the word embeddings (default: 64).
* **`hidden_dim`**: The hidden dimension used for the update layer (default: 128).
* **`num_concepts`**: The number of concepts in the knowledge bank (default: 100).
* **`reasoning_steps`**: The number of reasoning steps or iterations the model performs (default: 4).

---

## **Customizing the Model**

* **Knowledge Bank**: In this code, the knowledge bank is randomly initialized. You can modify this to load a pre-trained set of concept embeddings (e.g., from a knowledge graph or a semantic database).
* **Embedding Source**: The current model uses randomly initialized embeddings for the words in the sentence. It can be replaced with embeddings from models like **GloVe** or **BERT** for better semantic representations.

## **Future Improvements**

* **Pretrained Embeddings**: Integrate with GloVe, Word2Vec, or BERT embeddings for real-world tasks.
* **Dynamic Knowledge Bank**: Extend the knowledge bank to be dynamically updated or retrieved from external sources.
* **Task-specific Extensions**: Use the output tensor for tasks like **sentence classification**, **question answering**, or **semantic entailment**.

### ***Please feel free to provide constructive feedback!***
