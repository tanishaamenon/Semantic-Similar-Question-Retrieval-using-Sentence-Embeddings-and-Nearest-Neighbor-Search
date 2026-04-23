# Semantic Similar Question Retrieval using Sentence Embeddings and Nearest Neighbor Search

## Overview

This project implements a semantic retrieval system that retrieves questions similar in meaning to a user query using transformer embeddings and nearest neighbor search. The model uses the Quora Question Pairs dataset and supports interactive query-based retrieval along with evaluation metrics.

Example:

```text
Input Query:
How do I motivate myself for exams?

Retrieved Similar Questions:
- How can students stay motivated while studying?
- Ways to increase concentration during exam preparation
- How do I avoid procrastination before exams?
```

---

## Features

* Semantic question retrieval using Sentence Transformers
* Nearest Neighbor Search with cosine similarity
* Interactive user query input
* Jaccard similarity comparison between two queries
* Retrieval evaluation using:

  * Precision@5
  * Retrieval Accuracy
  * Average Cosine Similarity
  * Highest and Lowest Similarity

---

## Dataset

**Quora Question Pairs Dataset**

Downloaded using KaggleHub.

Contains question pairs from Quora spanning:

* Education
* Programming
* Health
* Finance
* Career
* General Knowledge

Subset used in project:

```python
10000 questions
```

---

## Methodology

### Step 1 — Download Dataset

```python
path = kagglehub.dataset_download("quora/question-pairs-dataset")
```

---

### Step 2 — Load Question Corpus

Questions from both columns are merged:

```python
questions = pd.concat(
[df["question1"], df["question2"]]
).dropna().head(10000).tolist()
```

---

### Step 3 — Generate Sentence Embeddings

Embedding model used:

```python
all-mpnet-base-v2
```

This converts questions into dense semantic vectors.

---

### Step 4 — Nearest Neighbor Retrieval

Retrieval model:

```python
nn = NearestNeighbors(
    n_neighbors=6,
    metric='cosine'
)
```

Given a query, the model retrieves top similar questions.

---

### Step 5 — Similarity Evaluation

Jaccard similarity:

```python
def jaccard_similarity(a,b):
    A=set(a.lower().split())
    B=set(b.lower().split())
    return len(A&B)/len(A|B)
```

Threshold used:

```python
threshold = 0.40
```

---

## Evaluation Metrics

Project evaluates retrieval performance using:

* Precision@5
* Retrieval Accuracy
* Average Cosine Similarity
* Relevant Results Count

Sample Output:

```text
------ MODEL EVALUATION ------

Precision@5: 0.80
Retrieval Accuracy: 80.0%
Average Cosine Similarity: 0.79
Highest Similarity: 0.86
Lowest Similarity: 0.68
Relevant Results: 4 out of 5
```

---

## Installation

```bash
pip install kagglehub pandas scikit-learn sentence-transformers
```

---

## Running the Notebook

Open:

```text
semantic_question_retrieval.ipynb
```

Run all cells and enter a query when prompted:

```text
Write a query to check other similar questions:
```

Example:

```text
How can I learn python quickly?
```

---

## Repository Structure

```text
semantic-question-retrieval/
│
├── semantic_question_retrieval.ipynb
├── semantic_question_retrieval.py
├── README.md
└── requirements.txt
```

---

## Example Queries to Test

```text
How do I motivate myself for exams?
How can I learn python quickly?
How do I improve concentration while studying?
What causes climate change?
```

---

## Applications

Possible applications include:

* Duplicate Question Detection
* Semantic Search Systems
* FAQ Retrieval
* Community Forum Search Optimization
* Intelligent Query Recommendation

---

## Course Relevance

Developed as a case study for:

**Machine Learning Clustering and Retrieval (CA3)**

Relevant topics covered:

* Retrieval
* Nearest Neighbor Search
* Semantic Similarity Search
* Approximate Similarity Thresholding

---

## Future Improvements

* Add Locality Sensitive Hashing (LSH)
* Scale retrieval using FAISS
* Deploy as a Streamlit web app
* Improve ranking using reranking models

---

## Course Relevance

Developed as a case study for: Machine Learning Clustering and Retrieval


---

## Author

Tanishaa Menon

---

## License

MIT License
