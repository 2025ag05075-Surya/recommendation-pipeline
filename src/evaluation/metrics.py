"""Ranking and rating metrics."""
import numpy as np


def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs, relevant, k):
    gains = np.array([1.0 if r in relevant else 0.0 for r in recs[:k]])
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    idcg = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(rec_idx, genre_dense):
    """1 - mean pairwise cosine similarity of the recommended items' genre vectors."""
    V = genre_dense[rec_idx]
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    S = V @ V.T
    n = len(rec_idx)
    return 1 - (S.sum() - np.trace(S)) / (n * (n - 1)) if n > 1 else 0.0


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
