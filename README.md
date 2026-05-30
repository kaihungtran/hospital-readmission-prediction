# Predicting 30-Day Hospital Readmission

**Catches 70% of patients who'd be readmitted within 30 days.** Multimodal neural network fusing ClinicalBERT-encoded discharge notes with structured ICU vitals and labs, trained on 50K+ admissions from MIMIC-IV. 13 model variants tested across 3 encoders. Best AUC: 0.703.

## Overview

Unplanned 30-day readmissions affect 1 in 5 Medicare patients and cost U.S. hospitals $17B annually. This project builds a discharge-time decision support tool that flags high-risk patients before they leave — giving care teams an actionable window to intervene.

We built a dual-branch neural network: one branch encodes free-text clinical notes using transformer embeddings (RoBERTa, SBERT, ClinicalBERT), the other processes structured features (vitals, lab results, diagnoses). The branches are fused and passed through a classification head trained on the binary readmission label.

## Results

| Model | Encoder | AUC | Recall |
|---|---|---|---|
| Dual-branch NN | RoBERTa | 0.681 | 0.63 |
| Dual-branch NN | SBERT | 0.689 | 0.66 |
| **Dual-branch NN + Attention + Chunking** | **ClinicalBERT** | **0.703** | **0.70** |

13 total variants tested (encoder type × pooling strategy × chunking approach).

## Architecture

```
Clinical Notes (free text)               Structured Features
        ↓                                  (vitals, labs, dx codes)
  ClinicalBERT                                     ↓
  (chunked, 512-token windows)             Dense layers
  Attention pooling → 768-dim                      ↓
        ↓                                          ↓
        └──────────── Fusion layer ────────────────┘
                            ↓
                    Classification head
                            ↓
                   Readmission probability
```

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_cleaning.ipynb` | Cohort construction, feature engineering, label creation |
| `02_clinical_bert_embeddings.ipynb` | ClinicalBERT embedding pipeline with chunking + attention pooling |
| `03_team_model_exploration.ipynb` | Full team exploration: baseline models, encoder comparisons |
| `04_baseline_nn.ipynb` | Baseline neural network (structured features only) |
| `05_sbert_model.ipynb` | Dual-branch NN with SBERT encoder |
| `06_clinicalbert_final.ipynb` | Final model — ClinicalBERT + attention + chunking (AUC 0.703) |

## Tech Stack

`Python` `Keras` `TensorFlow` `HuggingFace Transformers` `ClinicalBERT` `SBERT` `RoBERTa` `PySpark` `Google Cloud Platform` `MIMIC-IV`

## Data

This project uses [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/), a large de-identified ICU dataset requiring credentialed access through PhysioNet. We cannot redistribute the data. To reproduce, complete CITI training and apply for access at the link above. Data was processed via GCP (`gs://ba865-t9-mimiciv/`).

## Team

Built as part of **BA865 (Applied Machine Learning)** at Boston University with Purnima Khemka and Samuel Buelvas.

- Data cleaning & cohort construction: collaborative
- Embedding pipeline (ClinicalBERT chunking + attention): Kai Hung Tran
- Model variants 1–13: Kai Hung Tran
- Final model tuning: Kai Hung Tran

## Links

[Presentation](https://docs.google.com/presentation/d/11ByH8CnQc5nuK6h85FRMnA5D6SzX-SO_OQ1uZFXqidA/edit?usp=sharing)
