# 🧠 Toxic Comments Classification using Deep Learning

This project focuses on detecting toxic comments using Natural Language Processing (NLP) and Deep Learning models. We use Recurrent Neural Networks (LSTM, BiLSTM) and Dense Neural Networks to classify comments as toxic or non-toxic. The goal is to build a model that can help automatically moderate online conversations by identifying harmful or offensive content.

---

## 📌 Problem Statement

Online platforms often suffer from toxic and harmful user comments, which affect user experience and community standards. The objective of this project is to classify a given comment as **toxic** or **non-toxic** based on its content.

---

## 📂 Dataset

The dataset used is the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), which contains over 150,000 Wikipedia comments labeled for different types of toxicity:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

For simplicity, we labeled into a binary classification:
- **0**: Non-Toxic
- **1**: Toxic

---

## 🧹 Data Preprocessing

<p>I performed efficient text vectorization and batching using TensorFlow’s TextVectorization layer and tf.data pipeline to prepare the data for training</p>

```
vectorize = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode="int"
)

vectorized_text = vectorize(x.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

```

- Text Vectorization: Converts raw comment text into integer sequences.

- Output Length: Padded/truncated all sequences to a uniform length of 1800 tokens.

- Efficient Data Pipeline:

    - cache() for faster data access

    - shuffle() to randomize the order

    - batch() into batches of 16

    - prefetch() for optimized GPU/CPU usage during training
      
---

## 🏗️ Models Implemented

### 1️⃣ LSTM (Long Short-Term Memory)
- Embedding Layer
- LSTM Layer
- Dense Layers
- Output Layer (Sigmoid)

### 2️⃣ Bidirectional LSTM
- Embedding Layer
- Bidirectional LSTM
- Dense Layers
- Output Layer (Sigmoid)

### 3️⃣ Dense Neural Network
- TF-IDF Vectorization
- Dense Layers
- Dropout
- Output Layer (Sigmoid)

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

---

## ✅ Sample Results

| Model              | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| LSTM               | 0.93     | 0.89     | 0.95    |
| Bidirectional LSTM | 0.94     | 0.91     | 0.96    |
| Dense NN (TF-IDF)  | 0.89     | 0.85     | 0.92    |

> 📊 *Note: Values are sample placeholders. Replace them with your actual results.*

---

## 🔍 Example Predictions

| Comment                                              | Prediction |
|------------------------------------------------------|------------|
| "Thank you for your helpful feedback!"               | Non-Toxic  |
| "You're so dumb, this makes no sense."               | Toxic      |
| "I appreciate the detailed explanation."             | Non-Toxic  |
| "Nobody wants to hear your opinion, shut up."        | Toxic      |

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run training script
python train_model.py

# Step 3: Evaluate/Test
python evaluate_model.py
