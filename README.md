# Sequential Modelling Portfolio

Neural sequence modelling from scratch — RNN, LSTM, and transformer implementations with an incremental build approach. Culminates in mechanistic interpretability work using typoglycemic perturbations as probes — tracking cosine similarity of next predicted logits and WordNet-based word relatedness of next word predictions to locate where and how GPT-style models resolve surface-level distortions in learned representations.

---

## Overview

This portfolio documents a progressive, from-scratch journey through the core architectures of neural sequence modelling. Each project is self-contained and independently readable, but together they follow a deliberate arc: from understanding how recurrent networks process sequences, to implementing the attention mechanisms that replaced them, to using that implementation knowledge as the foundation for mechanistic interpretability research.

The three projects are ordered by conceptual dependency. The attention mechanism introduced in Project 1 is the direct precursor to the self-attention in Project 2. The transformer internals built in Project 2 ground the interpretability work in Project 3.

---

## Projects

### [01 — RNN to LSTM](https://github.com/your-username/seq-models-01-rnn-to-lstm)

An incremental build from a general-purpose vanilla RNN through to a full LSTM, with each step introducing a single architectural change against a fixed task.

**Progression:**
1. General-purpose RNN — trained and evaluated on a time series dataset
2. Naïve language model — one-to-many RNN with one-hot encoding, trained to sample novel text
3. Word embedding layer — replaces one-hot encoding; greedy search decoding
4. Beam search — replaces greedy decoding
5. Seq2seq — encoder component added
6. Bahdanau attention — all encoder hidden states connected to the decoder
7. LSTM cell — RNN cell replaced with LSTM

The structure is designed so that each step is a minimal delta on the previous one, keeping the variable of interest isolated.

---

### [02 — Transformers](https://github.com/your-username/seq-models-02-transformers)

From-scratch implementations of two transformer architectures: a full encoder-decoder transformer suited to seq2seq tasks, and a decoder-only (GPT-style) transformer for causal language modelling.

Built after Project 1 so that the move from Bahdanau attention to self-attention is motivated and concrete rather than abstract.

---

### [03 — Mechanistic Interpretability via Typoglycemic Probing](https://github.com/your-username/seq-models-03-mechinterp-typoglycemia)

Mechanistic interpretability of GPT-style transformers using typoglycemic perturbations — probing where and how learned representations handle surface-level distortions, and whether next-word prediction behaviour is sensitive to input surface form.

Typoglycemia — the phenomenon where words with scrambled interior letters remain readable to humans — is used as a controlled experimental lever. Matched pairs of normal and typoglycemic text are passed through pre-trained GPT-2 scale models, and two behavioural metrics serve as detectors:

- **Cosine similarity of next predicted logits** — measures how similar the model's output distributions are for a typoglycemic input versus its plaintext equivalent
- **Word relatedness of next word predictions** — uses WordNet to assess whether the predicted next words for both input types are semantically related

These metrics are not endpoints. They are localisation tools. Where they show divergence or drift, the mechanistic interpretability work begins — probing tokenisation, token embeddings, attention heads, and MLP layers to ask where and why the model's behaviour differs.

---

## Repository Structure

```
sequential-modelling-portfolio/
├── README.md
├── 01-rnn-to-lstm/
│   ├── README.md
│   ├── 01-general-purpose-rnn/
│   ├── 02-naive-language-model/
│   ├── 03-word-embeddings/
│   ├── 04-beam-search/
│   ├── 05-seq2seq/
│   ├── 06-attention/
│   └── 07-lstm/
├── 02-transformers/
│   ├── README.md
│   ├── encoder-decoder/
│   └── decoder-only/
└── 03-mechinterp-typoglycemia/
    ├── README.md
    ├── data/
    ├── notebooks/
    └── src/
```

Each project subdirectory contains its own README that stands alone. A visitor landing on any subfolder directly should be able to understand that project without context from this root.

---

## Branches

Active development follows one branch per project:

| Branch | Project |
|---|---|
| `main` | Stable, completed work only |
| `dev-rnn-to-lstm` | Project 1 |
| `dev-transformers` | Project 2 |
| `dev-mechinterp` | Project 3 |

Completed steps are merged into `main` when clean.

---


## Standalone Repositories

Each project is also published as a standalone repository as work is completed:

- [seq-models-01-rnn-to-lstm](https://github.com/Dorcas-Joy-Kahunguka/seq-models-01-rnn-to-lstm.git)
- [seq-models-02-transformers](https://github.com/Dorcas-Joy-Kahunguka/seq-models-02-transformers.git)
- [seq-models-03-mechinterp-typoglycemia](https://github.com/Dorcas-Joy-Kahunguka/seq-models-03-mechinterp-typoglycemia.git)

This monorepo is the working environment and carries the portfolio narrative. The standalone repos receive completed work and are independently readable.

