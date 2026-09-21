# Preserved academic artifacts

These files are university-era work, preserved via Git moves from `main` at `7e7b347deea08fc51f3f55b136950f75f4df2d7a`. They are not the canonical modern package. Running them may require missing model artifacts, obsolete relative paths, old dependencies, downloads, network access, or other unreproduced state. Their old numerical results are not currently validated benchmarks.

Moved from the repository root without rewriting their algorithms:

| Files | Historical role |
| --- | --- |
| `baseline_seq2seq.py` | Attention-based Seq2Seq baseline |
| `app.py`, `app_seq2seq.py`, `app_rag.py` | Streamlit interfaces |
| `evaluate.py`, `evaluate_seq2seq.py`, `evaluate_rag.py` | Evaluation scripts |
| `transformer_viz.py`, `trasnformer_rag_viz.py` | Visualization scripts (original spelling retained) |
| `utils.py` | Historical loading and row-random splitting |
| `Transformers_streamlit_colab.ipynb` | Academic notebook |
| `requirements.txt` | Historical dependency list; not the M0 environment |

The historical branches are separate experimental workspaces, not alternate modern-package releases:

- `Haotian-branch` at `16c0d680a6627b2c6aa2d34e0b232d68314b03c6` contains CodeT5/transformer work, `Transformer_loRA.py` using PEFT LoRA, and `Transformer_loRA_with optuna.py` using Optuna. Its scripts show model initialization/downloads at import time.
- `Vineeth-branch` at `a11383bbfd4b142d2597e1c56961a244cce8caf8` contains `Seq2seq_glove.py`, `loRA.py`, `transformer_rag.py`, `transformer_rag_FAISS.py`, `transformer_rag_threshold.py`, and `data_preproc.py`. These document GloVe, LoRA, retrieval, FAISS, threshold-based retrieval, and historical dataset construction. The branch also holds `600kdataset_table.csv` and a model file; neither is part of M0 on `main`.

Credit for the academic artifacts remains with their original contributors and Git history. M0 does not claim the old experiments are reproducible or their metrics valid.
