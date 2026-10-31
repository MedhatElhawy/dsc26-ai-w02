"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def _is_missing(val):
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    for col in [target, id_col, time_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    n_rows = len(df)
    
    # Missing fraction per column
    missing = {}
    for col in df.columns:
        miss_count = sum(1 for v in df[col] if _is_missing(v))
        missing[col] = float(miss_count / n_rows) if n_rows > 0 else 0.0

    # Duplicate IDs (non-missing id_col values appearing more than once)
    non_missing_ids = [v for v in df[id_col] if not _is_missing(v)]
    id_counts = pd.Series(non_missing_ids).value_counts()
    duplicate_ids = sorted(list(id_counts[id_counts > 1].index))

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Constant columns (other than target, at most one distinct non-missing value)
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        non_miss_vals = [v for v in df[col] if not _is_missing(v)]
        unique_non_miss = set(non_miss_vals)
        if len(unique_non_miss) <= 1:
            constant_columns.append(col)
    constant_columns = sorted(constant_columns)

    # Unlabelled rows (target is missing)
    unlabelled_mask = df[target].apply(_is_missing)
    unlabelled_rows = int(unlabelled_mask.sum())

    # Positive rate on labelled rows
    labelled_df = df[~unlabelled_mask]
    if len(labelled_df) > 0:
        y_labelled = labelled_df[target].astype(float)
        positive_rate = float((y_labelled == 1).mean())
    else:
        positive_rate = 0.0

    # Bad timestamps
    time_series = df[time_col]
    bad_timestamps = 0
    parsed_times = []
    for v in time_series:
        if _is_missing(v):
            parsed_times.append(pd.NaT)
            continue
        parsed = pd.to_datetime(v, errors="coerce")
        if pd.isna(parsed):
            bad_timestamps += 1
            parsed_times.append(pd.NaT)
        else:
            parsed_times.append(parsed)

    valid_times = [t for t in parsed_times if not pd.isna(t)]
    if valid_times:
        min_t = min(valid_times).isoformat()
        max_t = max(valid_times).isoformat()
        time_range = [min_t, max_t]
    else:
        time_range = [None, None]

    # Leakage suspects
    leakage_suspects = []
    if len(labelled_df) >= 20:
        y_arr = labelled_df[target].astype(float).values
        unique_classes = np.unique(y_arr)
        if len(unique_classes) == 2:
            for col in df.columns:
                if col in [target, id_col, time_col]:
                    continue
                # Single distinct value check (including missing as a value)
                col_vals_all = labelled_df[col].tolist()
                col_vals_normalized = [None if _is_missing(v) else v for v in col_vals_all]
                if len(set(col_vals_normalized)) <= 1:
                    continue

                # Check (a): numeric or boolean, at least 20 non-missing labelled values
                s_col = labelled_df[col]
                is_num_bool = pd.api.types.is_numeric_dtype(s_col) or pd.api.types.is_bool_dtype(s_col)
                if is_num_bool:
                    valid_mask = ~s_col.apply(_is_missing)
                    if valid_mask.sum() >= 20:
                        x_sub = s_col[valid_mask].astype(float).values
                        y_sub = y_arr[valid_mask]
                        if len(np.unique(y_sub)) == 2 and len(np.unique(x_sub)) > 1:
                            try:
                                auc = roc_auc_score(y_sub, x_sub)
                                if auc >= 0.98 or auc <= 0.02:
                                    if col not in leakage_suspects:
                                        leakage_suspects.append(col)
                            except Exception:
                                pass

                # Check (b): missingness indicator (1 = missing, 0 = not missing)
                missing_ind = labelled_df[col].apply(_is_missing).astype(int).values
                if len(np.unique(missing_ind)) == 2:
                    try:
                        auc_miss = roc_auc_score(y_arr, missing_ind)
                        if auc_miss >= 0.98 or auc_miss <= 0.02:
                            if col not in leakage_suspects:
                                leakage_suspects.append(col)
                    except Exception:
                        pass

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": int(duplicate_rows),
        "constant_columns": constant_columns,
        "unlabelled_rows": int(unlabelled_rows),
        "positive_rate": float(positive_rate),
        "bad_timestamps": int(bad_timestamps),
        "time_range": time_range,
        "leakage_suspects": sorted(leakage_suspects)
    }