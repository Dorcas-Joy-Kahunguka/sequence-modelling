"""
behavioral_utils.py

Utilities for the `behavioural-evaluation` notebook:
  - Step 2: cosine relatedness between predicted words
  - Step 3: WordNet relatedness between predicted words
  - Primary metric: classify clean/scrambled pairs by whether the two
    predictions are identical / related / unrelated in meaning to each
    other. This is what "resolving the surface distortion" means and does
    not require ground truth.
  - Diagnostics (secondary, ground-truth based): a reliable-baseline filter
    (was the clean prediction actually right, so any divergence found isn't
    just inherent sentence ambiguity?) and a trivial-agreement flag (is the
    model just defaulting to a generic word regardless of input?).

Requires: torch, nltk (with 'wordnet' and 'omw-1.4' corpora downloaded).

    import nltk
    nltk.download("wordnet")
    nltk.download("omw-1.4")

All functions that need a model expect a TransformerLens `HookedTransformer`
instance (they use `model.to_tokens` and `model.W_E`).
"""

import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# Step 2: cosine relatedness
# ---------------------------------------------------------------------------

def token_embedding(model, word, prepend_space=True):
    """
    Look up the input-embedding vector (row of W_E) for `word` under the
    model's own vocabulary/tokenizer.

    Words that BPE-split into multiple subtokens are handled by mean-pooling
    the embeddings of their subtokens, so this works for rare / multi-token
    words too, not just single-token ones.

    Returns a 1D torch tensor of shape (d_model,).
    """
    text = (" " + word.strip()) if prepend_space else word.strip()
    token_ids = model.to_tokens(text, prepend_bos=False)[0]
    embeds = model.W_E[token_ids]  # (n_subtokens, d_model)
    return embeds.mean(dim=0)


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two 1D torch tensors -> python float."""
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()


def word_cosine_relatedness(model, word_a, word_b):
    """Cosine similarity between two words' embeddings, under `model`."""
    va = token_embedding(model, word_a)
    vb = token_embedding(model, word_b)
    return cosine_similarity(va, vb)


# ---------------------------------------------------------------------------
# Step 3: WordNet relatedness
# ---------------------------------------------------------------------------

def _best_synset(word):
    """First (most frequent sense) synset for a word, or None if unknown."""
    synsets = wn.synsets(word.strip().lower())
    return synsets[0] if synsets else None


def wordnet_relatedness(word_a, word_b, metric="wup"):
    """
    Semantic relatedness between two words using WordNet.

    metric:
      "wup"  - Wu-Palmer similarity (0-1, taxonomy-depth aware). Default.
      "path" - Path similarity (0-1, 1/(shortest path length + 1)).

    Returns None if either word has no synset (punctuation, OOV, stray
    subword fragment, etc.) so callers can decide how to treat missing
    values (e.g. drop from aggregate stats) rather than silently getting 0.
    """
    syn_a = _best_synset(word_a)
    syn_b = _best_synset(word_b)
    if syn_a is None or syn_b is None:
        return None
    if metric == "path":
        return syn_a.path_similarity(syn_b)
    return syn_a.wup_similarity(syn_b)


# ---------------------------------------------------------------------------
# PRIMARY metric: consistency between clean and scrambled predictions
# ---------------------------------------------------------------------------
#
# The core hypothesis here is about invariance, not correctness: clean and
# scrambled sentences are meaning-equivalent, so a model that "resolves" the
# surface distortion should land on the same (or a near-synonymous) word
# either way, regardless of whether that word happens to match the original
# author's exact choice. Ground truth is intentionally *not* part of this
# classification - see the diagnostics section below for what it's used for
# instead.

CONSISTENCY_COSINE_THRESHOLD = 0.7
CONSISTENCY_WUP_THRESHOLD = 0.8


def classify_consistency(clean_word, scrambled_word, cosine_sim=None, wordnet_sim=None):
    """
    Categorize a pair by how similar the clean and scrambled predictions are
    to *each other*. This is the primary signal for Step 4: it directly
    answers "did the model resolve the surface distortion?" without
    reference to any ground truth.

      identical  - exact same predicted word/lemma
      related    - different word, but cosine and/or WordNet similarity
                   clears the threshold (resolved to the same meaning)
      unrelated  - neither embedding nor WordNet similarity clears the
                   threshold (distortion was not resolved)

    Pass in `cosine_sim` / `wordnet_sim` if already computed elsewhere to
    avoid recomputing; otherwise this only checks exact match.
    """
    if clean_word.strip().lower() == scrambled_word.strip().lower():
        return "identical"

    cosine_ok = cosine_sim is not None and cosine_sim >= CONSISTENCY_COSINE_THRESHOLD
    wordnet_ok = wordnet_sim is not None and wordnet_sim >= CONSISTENCY_WUP_THRESHOLD
    if cosine_ok or wordnet_ok:
        return "related"
    return "unrelated"


def summarize_consistency(records):
    """
    Given a list of per-pair result dicts containing a 'consistency_category'
    field, return counts per category.
    """
    from collections import Counter
    return dict(Counter(r["consistency_category"] for r in records))


# ---------------------------------------------------------------------------
# DIAGNOSTIC use of ground truth (not the primary metric)
# ---------------------------------------------------------------------------
#
# Ground truth is useful here for two narrower purposes:
#   1. Filtering to a "reliable baseline" subset - pairs where the clean
#      prediction actually matches truth - so that consistency results
#      aren't confounded by sentences that are inherently ambiguous even
#      without scrambling.
#   2. Flagging "trivial" agreement - cases where clean and scrambled agree
#      only because the model defaulted to a generic, low-information word
#      regardless of what it read, rather than because it genuinely
#      resolved the scrambled input.

def score_against_truth(model, predicted_word, true_word):
    """
    Bundle of relatedness scores between a model's predicted word and the
    actual (ground-truth) last word of the sentence. Used for the
    diagnostics below, not for the primary consistency classification.
    """
    return {
        "exact_match": predicted_word.strip().lower() == true_word.strip().lower(),
        "cosine": word_cosine_relatedness(model, predicted_word, true_word),
        "wordnet_wup": wordnet_relatedness(predicted_word, true_word, "wup"),
        "wordnet_path": wordnet_relatedness(predicted_word, true_word, "path"),
    }


def reliable_baseline_mask(records, require_exact=True, cosine_threshold=0.7):
    """
    Returns a list of booleans flagging which records have a trustworthy
    clean-condition baseline (clean prediction matches truth exactly, or -
    if require_exact=False - is at least close by cosine similarity).

    Restrict consistency analysis to `mask=True` records to rule out
    "the sentence itself was ambiguous even unscrambled" as an explanation
    for any clean/scrambled divergence you find.
    """
    mask = []
    for r in records:
        if require_exact:
            mask.append(bool(r.get("clean_correct")))
        else:
            cos = r.get("clean_vs_truth_cosine")
            mask.append(bool(r.get("clean_correct")) or (cos is not None and cos >= cosine_threshold))
    return mask


def flag_trivial_agreement(records, min_share=0.15):
    """
    Flags predicted words that show up as the clean-condition top-1
    prediction unusually often across the dataset (default: in >=15% of
    pairs) - a sign the model may be defaulting to a generic completion
    rather than actually reading the sentence. Pairs whose prediction is
    "identical" or "related" *and* involve a flagged generic word are more
    likely trivial agreement than genuine distortion-resolution.

    Returns (records_with_flag, flagged_words) where each record gains a
    'trivial_agreement_suspect' bool field.
    """
    from collections import Counter
    n = len(records)
    if n == 0:
        return records, set()

    counts = Counter(r["clean_top1"].strip().lower() for r in records)
    flagged_words = {w for w, c in counts.items() if c / n >= min_share}

    for r in records:
        word_is_generic = r["clean_top1"].strip().lower() in flagged_words
        is_agreeing = r.get("consistency_category") in ("identical", "related")
        r["trivial_agreement_suspect"] = bool(word_is_generic and is_agreeing)

    return records, flagged_words