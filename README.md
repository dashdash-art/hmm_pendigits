# Handwriting Recognition Using Hidden Markov Models (HMM)

This project implements a handwriting recognition system using **Hidden Markov Models** (HMMs) trained on the **Pendigits dataset**.

### 🚀 What It Does
- Learns handwritten digits (0–9) based on pen stroke coordinates
- Uses one HMM per digit, trained on sequences of motion (dx, dy)
- Predicts digits from unseen handwriting samples
- Reaches up to **95.28% accuracy**

---

### 📁 Dataset
**Source:** UCI Machine Learning Repository – Pen-Based Recognition of Handwritten Digits  
Files used:
- `pendigits.tra` – Training data
- `pendigits.tes` – Test data

Each sample includes 8 (x, y) pen points → reshaped into motion vectors for sequence modeling.

---

### 🛠️ How It Works
1. Loads training and test data from `.tra` and `.tes` files
2. Standardizes (scales) features
3. Converts absolute points into **motion vectors** (`dx`, `dy`)
4. Trains **10 Gaussian HMMs** (one per digit), each with 15 hidden states
5. Evaluates predictions on the test set

---

### 🧪 Final Accuracy
```bash
Accuracy: 95.28%
