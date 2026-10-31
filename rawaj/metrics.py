"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import math

def _validate_inputs(y_true, scores):
    y_arr = np.array(y_true)
    s_arr = np.array(scores, dtype=float)
    
    if len(y_arr) != len(s_arr):
        raise ValueError("y_true and scores must have the same length.")
    if len(y_arr) == 0:
        raise ValueError("Inputs cannot be empty.")
    for y in y_arr:
        if y not in [0, 1] and y not in [0.0, 1.0]:
            raise ValueError("Labels must be 0 or 1.")
    for s in s_arr:
        if math.isnan(s):
            raise ValueError("Scores cannot contain NaN.")
    return y_arr, s_arr

def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y_arr, s_arr = _validate_inputs(y_true, scores)
    flagged = s_arr >= threshold
    tp = int(np.sum((flagged) & (y_arr == 1)))
    fp = int(np.sum((flagged) & (y_arr == 0)))
    fn = int(np.sum((~flagged) & (y_arr == 1)))
    tn = int(np.sum((~flagged) & (y_arr == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    conf = confusion_at(y_true, scores, threshold)
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return (precision, recall)

def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    y_arr, s_arr = _validate_inputs(y_true, scores)
    n = len(y_arr)
    k = min(k, n)
    indices = np.argsort(-s_arr, kind='stable')
    top_k_y = y_arr[indices[:k]]
    return float(np.mean(top_k_y))

def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    conf = confusion_at(y_true, scores, threshold)
    cost = conf["fp"] * cost_fp + conf["fn"] * cost_fn
    return float(cost)

def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y_arr, s_arr = _validate_inputs(y_true, scores)
    candidates = sorted(list(np.unique(s_arr))) + [math.inf]
    
    best_t = None
    min_cost = float('inf')
    
    for t in candidates:
        cost = expected_cost(y_arr, s_arr, t, cost_fp=cost_fp, cost_fn=cost_fn)
        if cost < min_cost:
            min_cost = cost
            best_t = t
        elif cost == min_cost:
            if best_t is None or t > best_t:
                best_t = t
    return float(best_t)

def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y_arr, s_arr = _validate_inputs(y_true, scores)
    candidates = sorted(list(np.unique(s_arr)))
    
    valid_thresholds = []
    for t in candidates:
        p, r = precision_recall_at(y_arr, s_arr, t)
        if p >= min_precision:
            valid_thresholds.append(t)
            
    if not valid_thresholds:
        return None
    return float(min(valid_thresholds))