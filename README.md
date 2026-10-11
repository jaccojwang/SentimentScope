# SentimentScope: Binary Sentiment Analysis with Custom Transformers

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
**SentimentScope** is a natural language processing (NLP) project focused on binary sentiment classification of movie reviews using the **IMDB Dataset**. Instead of relying on off-the-shelf high-level wrappers, this project demonstrates custom PyTorch data loading pipelines and a transformer-based classifier (`DemoGPT`) using sequence mean pooling and a fully connected classification head.

---

## Performance Highlights
* **Task:** Binary Sentiment Classification (Positive / Negative)
* **Dataset:** 25,000 training & 25,000 testing movie reviews (IMDB)
* **Test Accuracy:** **76.60%** (exceeding the project target threshold of >75%)

---

## Key Features
- **Custom PyTorch Dataset & DataLoader:** Custom tokenization and sequence padding handling.
- **Transformer Architecture (`DemoGPT`):** Implements multi-head attention and position-wise feed-forward layers.
- **Mean Pooling Classification Head:** Aggregates transformer contextual embeddings across sequence length prior to linear layer classification.
- **GPU-Accelerated Training:** Optimized training and evaluation loops with cross-entropy loss and Adam optimizer.

---

## Project Structure
```text
├── SentimentScope_starter_Donaldo_Omondi.ipynb   # Main Jupyter Notebook (Data loading, Model, Training, Test)
├── README.md                                     # Project Documentation
└── .gitignore                                    # Excluded temporary files & checkpoints
