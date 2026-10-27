# Sarcasm Detection System

An end-to-end Natural Language Processing (NLP) system and Flask web application that classifies text comments as either **Sarcastic** or **Not Sarcastic**. This project evaluates and compares classical machine learning approaches (TF-IDF with Random Forest and AdaBoost) against a deep sequence model (Bidirectional LSTM) trained on Reddit comments.

---

## Overview

Sarcasm presents unique challenges for natural language understanding because it relies heavily on context, tonal contrast, and implicit meaning rather than explicit negative or positive sentiment.

This project benchmarks three different modeling strategies:

1. **Random Forest Classifier**: Evaluates lexical tokens using the top 5,000 TF-IDF word features across an ensemble of 50 decision trees.
2. **AdaBoost Classifier**: Evaluates an iterative boosting pipeline using 50 shallow decision stumps on 2,500 TF-IDF features.
3. **Bidirectional LSTM Neural Network**: Models sequence-level word dependencies using a 10,000-word vocabulary, 16-dimensional embedding layer, and 32 bidirectional recurrent units with early stopping.

A lightweight Flask web application is provided to serve real-time predictions and display model confidence.

---

## System Architecture

```text
                           User Input Text
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
TF-IDF Vectorizer (5,000 features)          Keras Tokenizer & Padding (len=50)
         │                                                 │
         ▼                                                 ▼
Random Forest Classifier                          Bidirectional LSTM (32 units)
(50 Trees, Max Depth 20)                          + Dropout (0.5) + Sigmoid
         │                                                 │
         ▼                                                 ▼
Prediction: Sarcastic / Normal               Prediction + Confidence Probability (%)
         └────────────────────────┬────────────────────────┘
                                  ▼
                     Rendered in Flask Web UI

```

---

## Dataset & Preprocessing

The model is trained on a balanced subset of the **Self-Annotated Reddit Sarcasm Corpus** (`train-balanced-sarcasm.csv`).

* **Class Distribution:** 100,000 comments sampled uniformly (50,000 Sarcastic [`1`], 50,000 Non-Sarcastic [`0`]).
* **Text Preprocessing Pipeline:**
* Lowercasing text inputs.
* Stripping URLs (`http\S+`, `www\S+`), subreddits (`r/\w+`), mentions (`@\w+`), and hashtags (`#\w+`).
* Removing punctuation and numbers to focus exclusively on linguistic tokens.
* Normalizing whitespace and stripping blank samples.



---

## Model Performance

Models were evaluated on a held-out test split of 19,947 comments:

| Architecture | Representation | Key Hyperparameters | Test Accuracy |
| --- | --- | --- | --- |
| **AdaBoost** | TF-IDF (2,500 features) | Base: Decision Stump, $n=50$ | 56.83% |
| **Random Forest** | TF-IDF (5,000 features) | $n=50$, Max Depth: 20 | 65.18% |
| **Bidirectional LSTM** | Embedding (16-dim, length 50) | 32 Bi-LSTM Units, Dropout 0.5 | **69.74%** |

---

## Installation & Setup

### 1. Prerequisites

* Python 3.9+
* Git

### 2. Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd "Sarcasm Detection"

```

### 3. Create a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 5. Verify Model Artifacts

Ensure the following files are present in the root directory before running the server:

* `sarcasm_random_forest.joblib`
* `tfidf_vectorizer.joblib`
* `sarcasm_lstm.keras`
* `lstm_tokenizer.json`

*(Note: These files can be generated from scratch by running all cells in `sarcasm_detection_training.ipynb`.)*

---

## Running the Application

Launch the Flask server locally:

```bash
python flask_server.py

```

Access the interface by navigating to `http://localhost:5000` in your web browser.

To run with debug mode enabled (local development only):

```bash
# Windows (PowerShell)
$env:FLASK_DEBUG="true"
python flask_server.py

# Linux / macOS
export FLASK_DEBUG="true"
python flask_server.py

```