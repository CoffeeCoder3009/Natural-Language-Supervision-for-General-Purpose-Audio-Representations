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


### 2. Setup
First, install python 3.8 or higher (3.11 recommended). Then, install CLAP using either of the following:

```bash
# Install pypi pacakge
pip install msclap

# Or Install latest (unstable) git source
pip install git+https://github.com/microsoft/CLAP.git
```

### 3. Run inference on a dataset

```bash
python ZS_classification_<dataset>.py

# for example run python ZS_classification_esc_50.py
```


## Results on Datasets

| Dataset       | Zero-Shot Accuracy (ours) |  Zero-Shot Accuracy (actual) 
|---------------|---------------------|---------------------|
| ESC-50        | 87.75%(avg of all folds)              |  93.9%             |
| US8K          |       79.85%    |  82.3%             |
| FSD50K          |               |  0.485 (mAP)     |
   
## Contributors

- [Purvanshi Nijhawan](https://github.com/CoffeeCoder3009)
- [Asmi Srivastava](https://github.com/asmisriva)

