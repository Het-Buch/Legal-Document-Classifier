# src/infer/ensemble.py
"""
Small ensemble utilities.
- majority_vote: takes list of predictions (list of (N,L) arrays) and returns majority-voted preds
- weighted_average: takes list of probs arrays and weights, returns averaged probs
"""
import numpy as np

def majority_vote(preds_list):
    """
    preds_list: list of lists (N x L) predicted binary arrays
    returns: (N x L) binary majority-vote
    """
    if not preds_list:
        return []
    arrs = np.array(preds_list)  # (M, N, L)
    summed = arrs.sum(axis=0)
    M = arrs.shape[0]
    result = (summed > (M / 2)).astype(int)
    return result.tolist()

def weighted_average(probs_list, weights=None):
    """
    probs_list: list of (N,L) float arrays
    weights: list of length M
    """
    if not probs_list:
        return []
    arrs = np.array(probs_list)  # (M, N, L)
    M = arrs.shape[0]
    if weights is None:
        weights = np.ones(M) / M
    weights = np.array(weights) / float(np.sum(weights))
    avg = np.tensordot(weights, arrs, axes=(0,0))
    return avg.tolist()
