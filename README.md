# RAG From Scratch

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) pipeline from scratch** to demonstrate how modern AI systems reduce hallucinations in Large Language Models (LLMs).

Instead of relying only on a model’s internal knowledge, RAG systems retrieve relevant information from external documents and provide it as context to the model before generating an answer.

This repository shows **how RAG works internally**, including embeddings, vector similarity search, and context-grounded generation.

The project also compares responses **with and without retrieval** to clearly demonstrate the impact of RAG on answer quality.

---

# Key Idea

Large Language Models generate responses based on **probability distributions learned during training**, which can sometimes lead to **hallucinations** or incorrect answers when the model lacks specific information.

Retrieval-Augmented Generation solves this by:

1. Converting documents into **vector embeddings**
2. Storing those embeddings in a **vector database**
3. Retrieving the most relevant document chunks when a question is asked
4. Providing those chunks as **context to the LLM**

This allows the model to produce **grounded and accurate responses**.

---

# System Architecture

```
Document
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Database
   ↓
User Question
   ↓
Question Embedding
   ↓
Similarity Search
   ↓
Relevant Document Chunks
   ↓
Prompt Augmentation
   ↓
LLM Response
```

---

# Project Features

* Implementation of **RAG pipeline from scratch**
* **Document chunking and embedding generation**
* **Vector similarity search using FAISS**
* **Comparison of LLM responses with and without RAG**
* **Embedding visualization to illustrate semantic similarity**
* Simple and modular architecture for learning how RAG works internally

---

# Technologies Used

* Python
* Sentence Transformers
* FAISS (Facebook AI Similarity Search)
* Groq LLM API
* Matplotlib
* Scikit-learn

---

# Repository Structure

```
rag-from-scratch
│
├── data/
│   └── knowledge.txt
│
├── embedder.py
├── vector_store.py
├── rag_pipeline.py
├── no_rag_pipeline.py
├── visualize_embeddings.py
├── build_index.py
├── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How the System Works

## 1. Document Processing

The system begins with a knowledge document containing domain-specific information.

The document is split into smaller **chunks** to ensure efficient retrieval and better context management.

---

## 2. Embedding Generation

Each chunk is converted into a **vector embedding** using a sentence-transformer model.

These embeddings capture the **semantic meaning of text**, allowing similar concepts to appear closer in vector space.

---

## 3. Vector Storage

All embeddings are stored in a **FAISS vector index** for efficient similarity search.

FAISS enables fast retrieval of the most relevant chunks for a given query.

---

## 4. Query Processing

When a user asks a question:

1. The question is converted into an embedding
2. FAISS searches for the closest vectors
3. The most relevant document chunks are retrieved

---

## 5. Response Generation

The retrieved chunks are added as **context in the prompt** and sent to the LLM.

The model then generates an answer **grounded in the retrieved information**.

---

# RAG vs Non-RAG Comparison

## Without RAG

The model answers based only on its internal training data.

This can result in:

* Unknown answers
* Hallucinated information
* Incomplete responses

## With RAG

The system retrieves relevant information from the document before generating the answer.

Benefits:

* Context-aware responses
* Reduced hallucinations
* Domain-specific knowledge access

---

# Embedding Visualization

The project also includes a visualization that demonstrates how semantic similarity works.

Document chunks and user queries are embedded into high-dimensional vector space.
These vectors are projected into 2D space to illustrate how the system retrieves the closest chunks during similarity search.

This visualization helps explain how **vector databases enable semantic retrieval in RAG systems**.

---

# Example Query

Question:

```
What is Project Helios?
```

Without RAG:
The language model may not have knowledge about the topic.

With RAG:
The system retrieves relevant chunks from the document and produces a grounded answer based on the provided context.

---

# Installation

Clone the repository:

```
git clone https://github.com/Goutam2803/RAG-From-Scratch.git
```

Navigate to the project folder:

```
cd RAG-From-Scratch
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

Build the FAISS index:

```
python build_index.py
```

Run the RAG vs Non-RAG comparison:

```
python main.py
```

Visualize embedding similarity:

```
python visualize_embeddings.py
```

---

# Learning Goals

This project was built to better understand:

* Retrieval-Augmented Generation
* Vector embeddings
* Semantic similarity search
* Vector databases
* Grounded generation in LLM systems

---

# Future Improvements

Potential extensions for this project include:

* Web interface for asking questions
* Support for multiple documents
* Advanced chunking strategies
* Hybrid search (keyword + vector)
* Persistent vector databases
* Streamlit-based UI for interactive demonstrations

---

# Author

**Goutam Raj**

AI / Machine Learning Enthusiast
Focused on understanding the internal mechanics of modern AI systems.

---

# License

This project is open source and available for educational and research purposes.
