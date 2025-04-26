# Natural-Language-Supervision-for-General-Purpose-Audio-Representations

This repository contains an implementation and benchmarking framework for the **CLAP** model on various public audio datasets.

CLAP (Contrastive Language-Audio Pretraining) learns general-purpose audio representations directly from natural language supervision, enabling **zero-shot audio classification**, **retrieval**, and **captioning** across multiple domains.

---

## Paper Details

- **Title:** Natural Language Supervision for General-Purpose Audio Representations  
- **Authors:** Benjamin Elizalde, Soham Deshmukh, Huaming Wang  
- **arXiv:** [2309.05767](https://arxiv.org/abs/2309.05767)  

---

## How to Run

### 1. Clone the repo & install dependencies

```bash
git clone https://github.com/CoffeeCoder3009/Natural-Language-Supervision-for-General-Purpose-Audio-Representations.git
```


### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run inference on a dataset

```bash
python esc50_eval.py
```


## Results on Datasets

| Dataset       | Zero-Shot Accuracy (ours) |  Zero-Shot Accuracy (actual) 
|---------------|---------------------|---------------------|
| ESC-50        | 87.75%(avg of all folds)              |  93.9%             |
| US8K          |       79.85%    |  82.3%             |
| FSD50K          |               |           |
   

