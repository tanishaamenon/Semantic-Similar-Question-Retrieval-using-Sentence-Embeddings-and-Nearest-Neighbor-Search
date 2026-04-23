# Semantic Similar Question Retrieval using Sentence Embeddings and Nearest Neighbor Search

## Overview

This project implements a **semantic question retrieval system** that finds and retrieves questions similar in meaning to a user query using transformer-based sentence embeddings and nearest neighbor search.

Instead of relying on simple keyword matching, the system captures **semantic similarity**, allowing it to retrieve paraphrased questions with different wording but similar intent.

Example:

**Query:**

```text
How do I motivate myself for exams?
```

**Retrieved Similar Questions:**

```text
How can students stay motivated while studying?
Ways to increase concentration during exam preparation
How do I avoid procrastination before exams?
```

---

## Features

* Semantic Similarity Retrieval using Sentence-BERT embeddings
* Nearest Neighbor Search using cosine similarity
* Retrieval evaluation using Precision@K and similarity metrics
* Real-world case study using Quora Question Pairs dataset
* Interactive custom query testing

---

## Problem Statement

Online platforms often contain duplicate or semantically similar questions. Efficiently identifying and retrieving similar questions can:

* Reduce duplicate content
* Improve search efficiency
* Reuse existing answers
* Support intelligent question recommendation systems

This project addresses this using semantic retrieval techniques.

---

## Dataset

**Quora Question Pairs Dataset**

Source: Kaggle

Contains:

* Question pairs
* Duplicate/non-duplicate labels
* Large variety of domains:

  * Education
  * Technology
  * Health
  * Career
  * General Knowledge

---

## Methodology

### 1. Data Preprocessing

* Load question pairs dataset
* Merge question1 and question2
* Remove null values
* Select subset for indexing

### 2. Semantic Embedding Generation

Questions are converted into dense vector representations using:

```python
paraphrase-MiniLM-L6-v2
```

### 3. Nearest Neighbor Retrieval

A cosine similarity based nearest neighbor model retrieves the top-k most similar questions.

### 4. Evaluation

Model performance evaluated using:

* Precision@K
* Retrieval Accuracy
* Average Cosine Similarity

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Sentence Transformers
* KaggleHub

---

## Project Workflow

```text
User Query
   ↓
Sentence Embedding
   ↓
Nearest Neighbor Search
   ↓
Top-K Similar Questions Retrieved
```

---

## Sample Results

Example query:

```text
How can I learn python quickly?
```

Retrieved:

```text
How do I learn Python fast?
Best way to study Python quickly
How can I improve python skills?
```

Sample Evaluation:

```text
Precision@5 : 0.80
Average Cosine Similarity : 0.79
Retrieval Accuracy : 80%
```

---

## Installation

```bash
pip install kagglehub pandas scikit-learn sentence-transformers
```

---

## Run the Project

```bash
jupyter notebook
```

Run the notebook cells and test with custom queries:

```python
retrieve_similar("How do I improve concentration while studying?")
```

---

## Repository Structure

```text
semantic-question-retrieval/
│
├── Semantic_Retrieval.ipynb
├── README.md
└── requirements.txt
```

---

## Applications

Possible applications include:

* Duplicate question detection
* Semantic search engines
* FAQ retrieval systems
* Educational query recommendation
* Community forum search optimization

---

## Future Improvements

* Add Locality Sensitive Hashing (LSH)
* Scale using FAISS for large datasets
* Improve ranking using transformer reranking
* Deploy as a Streamlit web app

---

## Course Relevance

Developed as a case study for: Machine Learning Clustering and Retrieval


---

## Author

Tanishaa Menon

---

## License

MIT License
