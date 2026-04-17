<div align="center">

# Recommendations with IBM Watson Studio

**Article recommender system using collaborative filtering, SVD matrix factorization, and content-based methods**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/license-MIT-success?style=flat)](LICENSE)

</div>

---

## What This Project Demonstrates

A full recommendation system built on real user-article interaction data from the IBM Watson Studio platform — covering the main approaches used in production recommender systems, from the simplest to the most powerful.

---

## Business Problem

IBM Watson Studio has thousands of articles. Users interact with a subset. The challenge: **what articles should we recommend to each user** to maximize engagement and discovery?

This project implements and compares four recommendation strategies on the same dataset.

---

## Methods Implemented

### 1. Rank-Based Recommendations
Recommend the most popular articles globally — strong baseline for new users (cold-start problem). No user history needed.

### 2. User-Based Collaborative Filtering
Find users with similar interaction history, recommend articles they engaged with but the target user has not seen yet.

```
similarity(user_a, user_b) = dot product of binary interaction vectors
```

### 3. Content-Based Filtering
Recommend articles similar to those a user has already interacted with, using article title text features.

### 4. Matrix Factorization (SVD)
Decompose the full user-item interaction matrix using Singular Value Decomposition. Captures latent factors in user preferences.

```
R ≈ U · Σ · Vᵀ
```
- Evaluated by varying the number of latent factors (k)
- Analyzed train vs. test accuracy to detect overfitting

---

## Dataset

| Entity | Size |
|---|---|
| User-article interactions | 45,993 interactions |
| Unique users | 5,148 |
| Unique articles | 714 |

---

## Key Findings

- Rank-based: effective for cold-start, no personalization
- Collaborative filtering: improves with more user history
- SVD: higher k → better train accuracy but risks overfitting on test set
- Hybrid approach (rank + collaborative) handles both new and returning users

---

## Project Structure

```
Recommendations_with_IBM/
├── Recommendations_with_IBM.ipynb  # Main notebook — all 4 methods
├── user-item-interactions.csv      # Raw interaction data
├── articles_community.csv          # Article metadata
└── top_5/10/20.p                   # Precomputed top-N article lists
```

---

## Quickstart

```bash
git clone https://github.com/FabioCLima/Recommendations_with_IBM.git
cd Recommendations_with_IBM

pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Recommendations_with_IBM.ipynb
```

---

## Skills Demonstrated

`Collaborative filtering` · `Matrix factorization (SVD)` · `Content-based filtering` · `Cold-start problem` · `User-item matrices` · `NumPy` · `pandas` · `Recommender systems` · `Evaluation methodology`
