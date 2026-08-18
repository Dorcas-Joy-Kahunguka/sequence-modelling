
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from src.data_utils import load_cache_record

import pandas as pd
import torch







# --------------------------------------------------------------------------- #
# Tokenization comparison
# --------------------------------------------------------------------------- #

@dataclass
class TokenizationComparison:
    is_same: bool
    message: str
    clean_tokens: Any = None
    scrambled_tokens: Any = None
    clean_str_tokens: list = field(default_factory=list)
    scrambled_str_tokens: list = field(default_factory=list)


def _tokens_equal(clean_tokens: torch.Tensor, scrambled_tokens: torch.Tensor) -> bool:
    """
    Two token tensors "tokenize the same" iff they have the same shape and
    every token id matches (i.e. the same numerical values in the same
    positions). Different lengths automatically count as different.
    """
    clean_tokens = torch.as_tensor(clean_tokens).flatten()
    scrambled_tokens = torch.as_tensor(scrambled_tokens).flatten()
    if clean_tokens.shape != scrambled_tokens.shape:
        return False
    return bool(torch.equal(clean_tokens, scrambled_tokens))


def compare_tokenization(clean_tokens: torch.Tensor,
                          scrambled_tokens: torch.Tensor,
                          clean_str_tokens: list,
                          scrambled_str_tokens: list) -> TokenizationComparison:
    """
    Core comparison function. Compares numerical token ids for equality.
    If identical -> reports "tokenize the same".
    If not -> reports "tokenize differently" along with the str_tokens for
    both, formatted as: "clean tokenized to X and scramble tokenized to Y".
    """
    same = _tokens_equal(clean_tokens, scrambled_tokens)

    if same:
        message = "clean and scramble tokenize the same"
    else:
        message = (
            f"clean tokenized to {clean_str_tokens} "
            f"and scramble tokenized to {scrambled_str_tokens}"
        )

    return TokenizationComparison(
        is_same=same,
        message=message,
        clean_tokens=clean_tokens,
        scrambled_tokens=scrambled_tokens,
        clean_str_tokens=list(clean_str_tokens),
        scrambled_str_tokens=list(scrambled_str_tokens),
    )


def compare_tokenization_from_cache(cache_path: str | Path) -> TokenizationComparison:
    """
    Convenience wrapper: loads the .pt cache file at cache_path and runs
    compare_tokenization on the clean/scrambled tokens + str_tokens found
    inside it.
    """
    record = load_cache_record(cache_path)
    clean = record["clean"]
    scrambled = record["scrambled"]
    return compare_tokenization(
        clean_tokens=clean["tokens"],
        scrambled_tokens=scrambled["tokens"],
        clean_str_tokens=clean["str_tokens"],
        scrambled_str_tokens=scrambled["str_tokens"],
    )


# --------------------------------------------------------------------------- #
# Batch processing over the records dataframe
# --------------------------------------------------------------------------- #

def annotate_tokenization(df: pd.DataFrame,
                           cache_path_col: str = "cache_path",
                           verbose: bool = False) -> pd.DataFrame:
    """
    Iterates over every row of df, loads its cache file, compares clean vs
    scrambled tokenization, and returns a COPY of df with two new columns:

        tokenization_is_same : bool
        tokenization_message : str  (human readable, matches the
                                      "clean tokenized to X and scramble
                                      tokenized to Y" format when different)

    Rows whose cache file fails to load get tokenization_is_same = pd.NA and
    an explanatory message, rather than raising, so one bad file doesn't
    kill the whole batch.
    """
    is_same_col = []
    message_col = []

    for _, row in df.iterrows():
        cache_path = row[cache_path_col]
        try:
            comparison = compare_tokenization_from_cache(cache_path)
            is_same_col.append(comparison.is_same)
            message_col.append(comparison.message)
            if verbose:
                print(f"[id={row.get('id')}] {comparison.message}")
        except Exception as e:
            is_same_col.append(pd.NA)
            message_col.append(f"ERROR loading cache: {e}")
            if verbose:
                print(f"[id={row.get('id')}] ERROR: {e}")

    out = df.copy()
    out["tokenization_is_same"] = is_same_col
    out["tokenization_message"] = message_col
    return out


def save_records(df: pd.DataFrame,
                  csv_path: Optional[str | Path] = None,
                  json_path: Optional[str | Path] = None) -> None:
    """Save the annotated records back out to CSV and/or JSON."""
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    if json_path is not None:
        df.to_json(json_path, orient="records", indent=2)


# --------------------------------------------------------------------------- #
# Confusion matrix: consistency_category x tokenization sameness
# --------------------------------------------------------------------------- #

def tokenization_confusion_matrix(df: pd.DataFrame,
                                   consistency_col: str = "consistency_category",
                                   is_same_col: str = "tokenization_is_same") -> pd.DataFrame:
    """
    Builds a matrix with consistency categories as columns and
    tokenization_is_same / tokenization_is_different as rows, containing
    counts of records in each (category, sameness) bucket.

    Rows that failed to load (NA in is_same_col) are excluded from the
    matrix but a warning count is printed.
    """
    n_missing = df[is_same_col].isna().sum()
    if n_missing:
        print(f"Note: {n_missing} record(s) had no tokenization comparison "
              f"(cache load failed) and are excluded from the matrix.")

    valid = df.dropna(subset=[is_same_col]).copy()
    valid["_tok_label"] = valid[is_same_col].map(
        {True: "tokenization_is_same", False: "tokenization_is_different"}
    )

    matrix = pd.crosstab(valid["_tok_label"], valid[consistency_col])

    # Ensure both rows always present, even if a category has 0 in one bucket
    for label in ["tokenization_is_same", "tokenization_is_different"]:
        if label not in matrix.index:
            matrix.loc[label] = 0
    matrix = matrix.reindex(["tokenization_is_same", "tokenization_is_different"])

    # Ensure consistent, expected category ordering when present
    expected_order = ["identical", "related", "unrelated"]
    ordered_cols = [c for c in expected_order if c in matrix.columns]
    ordered_cols += [c for c in matrix.columns if c not in expected_order]
    matrix = matrix[ordered_cols]

    matrix["total"] = matrix.sum(axis=1)
    matrix.loc["total"] = matrix.sum(axis=0)

    return matrix


