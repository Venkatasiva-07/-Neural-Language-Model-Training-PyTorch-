# Neural Language Model — Assignment 2


## Overview
Word-level LSTM language model implemented from scratch using PyTorch. This repository contains code, notebooks, model checkpoints (linked separately via Google Drive), plots, and a short report demonstrating underfitting, overfitting, and best-fit experiments.


## Repository Structure


## 🚀 Overview  
This project implements a **Neural Language Model from scratch** using **PyTorch**.  
A **word-level LSTM** is trained on the provided dataset to predict the next word in a sequence.

This assignment demonstrates:

- ✔️ Underfitting  
- ✔️ Overfitting  
- ✔️ Best-fit model  
- ✔️ Training & validation loss graphs  
- ✔️ Perplexity evaluation  
- ✔️ Text generation samples  

---

## 📄 Dataset  
- The dataset used is:  
  **Pride and Prejudice — Jane Austen**  
- Tokenization: **Whitespace tokenization** 25000 
- Preprocessing:  
  - Lowercasing  
  - Removing newline characters  
  - Removing extra spaces  

---
## 🧠 Model Architecture (LSTM)

```
Embedding Layer (word-level)
→ 2-layer LSTM
→ Fully Connected Layer (vocab projection)
→ Softmax (through CrossEntropyLoss)
```

Hyperparameters:  
```
Embedding Size: 128
Hidden Size: 256
Sequence Length: 50
Batch Size: 64
Optimizer: Adam
Loss: CrossEntropyLoss
Gradient Clipping: 1.0
Early Stopping: patience = 3
```

---

## 🧪 Experiments

### ✔️ **1. Underfitting Model**
- embed_dim = 128  
- hidden_dim = 64  
- num_layers = 1  
- epochs = 3  

### ✔️ **2. Overfitting Model**  
- embed_dim = 256  
- hidden_dim = 512  
- num_layers = 3  
- tiny dataset slice (3000 tokens)

### ✔️ **3. Best-Fit Model**  
- embed_dim = 128  
- hidden_dim = 256  
- num_layers = 2  
- full dataset

---

## 📊 Training vs Validation Loss Plots  
All plots are saved in Google Drive:

👉 **Drive Folder:**  
`https://drive.google.com/drive/folders/PUT_YOUR_FOLDER_LINK_HERE`

Files included:  
- `underfit_loss.png`  
- `overfit_loss.png`  
- `bestfit_loss.png`  
- `combined_loss_plot.png`  


 ---

## 📉 Final Metrics

| Model     | Val Loss | Perplexity |
|-----------|----------|------------|
| Underfit  | 6.9171   | 1009.38    |
| Overfit   | 9.0280   | 8333.23    |
| Best-Fit  | 7.3997   | 1635.63    |

---

## ✍️ Sample Text Generation

**Prompt:** `Elizabeth`  
**Output (Best Model):**  
```
elizabeth had been a very time which he is not for a few of the whole man as her sister was not at the whole in the letter was
```

---

## ▶️ How to Run (Google Colab)

```
from google.colab import drive
drive.mount('/content/drive')

# Open notebook.ipynb and run all cells
```

OR run script:

```
python train_language_model.py --data_path <path> --out_dir ./out
```

---

## 📦 Requirements

```
torch
numpy
pandas
matplotlib
```

---

## 📝 Report  
The full detailed report is available in **report.md** inside this repository.

---

## 📬 Contact  
venkata siva ch  
Email: venkatasivach15@gmail.com

---

# ⭐ Notes  
- Model checkpoints (.pth) are stored in Google Drive due to size limits.  


