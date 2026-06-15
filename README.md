# Multilingual Health Question Answering — Low-Resource African Languages

Final project for Machine Learning Techniques I. Zindi competition: *Multilingual
Health Question Answering in Low-Resource African Languages Challenge*.

## Task

Given a maternal/sexual/reproductive health question in one of five languages
(English, Akan, Luganda, Swahili, Amharic), generate a fluent, accurate answer
in the same language. Evaluated on ROUGE-1 F1, ROUGE-L F1, and LLM-as-a-Judge.

## Repository structure

```
.
├── README.md                     <- this file
├── requirements.txt              <- Python package dependencies
├── notebooks/
│   └── training_pipeline.ipynb   <- single master notebook, all experiments
│                                     run from this by changing config variables
├── src/                          <- reusable functions extracted from notebook
│   ├── data_utils.py
│   ├── retrieval.py
│   ├── finetune.py
│   └── submission.py
├── docs/
│   └── EXPERIMENT_LOG.md         <- full experiment tracking table (all 10 experiments)
├── submissions/
│   └── (generated CSVs — not committed, see .gitignore)
└── report/
    └── Dushime_Paulette_FinalProject.pdf   <- full written report
```

## How to reproduce

1. Open `notebooks/training_pipeline.ipynb` in Google Colab (badge below) or Kaggle.
2. If using Kaggle: attach the multilingual dataset and update DATA_DIR in the Paths cell.
   If using Colab: mount Google Drive and upload Train.csv, Test.csv, Val.csv,
   SampleSubmission.csv to My Drive/multilingual_health_qa/.
3. Confirm GPU is attached and CUDA is available — run the verification cell near
   the top before anything else:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
4. Run all cells in order (Runtime → Run All). The notebook is structured so every
   experiment is controlled by a small set of config variables (MODEL_NAME,
   FINETUNE_EPOCHS, EMBED_MODEL_NAME, SIMILARITY_THRESHOLD, etc.) — change these
   to reproduce a specific experiment from docs/EXPERIMENT_LOG.md.
5. The final cells generate submission_*.csv files, formatted with the required
   ID, TargetRLF1, TargetR1F1, TargetLLM columns (all three target columns
   identical, per competition rules).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
<!-- replace # with the actual Colab share link once notebook is uploaded to GitHub -->

## Approach summary

Two approaches were developed and compared across 10 experiments
(full details in docs/EXPERIMENT_LOG.md):

- **Generative fine-tuning:** google/mt5-small fine-tuned with HuggingFace
  Seq2SeqTrainer (Adafactor optimizer, for GPU memory efficiency).
- **Semantic retrieval (best approach):** Multilingual sentence embeddings
  (sentence-transformers/LaBSE) used to retrieve the most similar training
  question's answer via cosine similarity, restricted to the same language subset.
  The final submission uses a combined train+validation retrieval pool
  (~36,500 candidate questions).

**Best result:** weighted leaderboard score of 0.5483 (ROUGE-1 F1 0.5377,
ROUGE-L F1 0.4677, LLM-as-a-Judge 0.6782), a 99% relative improvement over
the initial fine-tuned baseline (0.2749).

See docs/EXPERIMENT_LOG.md for the full progression of experiments, including
infrastructure debugging (GPU memory, multi-GPU handling, environment/CUDA
mismatches) that shaped the final configuration, and
report/Dushime_Paulette_FinalProject.pdf for the full written analysis.

## Known constraints

- Trained and evaluated on a single NVIDIA T4 GPU (16GB) via Kaggle/Colab free tier.
  Batch size, sequence length, and optimizer choice were tuned specifically for this
  hardware ceiling — see experiment log for details.
- Fine-tuning model: google/mt5-small. Larger multilingual models
  (facebook/nllb-200-distilled-600M, google/mt5-base) were evaluated but required
  additional memory optimization to run reliably on this hardware.
- Retrieval: the best-performing approach (LaBSE-based semantic retrieval) can only
  return answers present in its training/validation candidate pool — it cannot
  generate genuinely novel answers to unfamiliar questions. See the report's
  Discussion section for the full tradeoff analysis.

## Author

Paulette Dushime — BSc Software Engineering (ML), African Leadership University
