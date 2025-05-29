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
- Dense Layers
- Output Layer (Sigmoid)

```
model = Sequential()
model.add(Input(shape=(1800,)))  # Explicit input shape
# Create the embedding layer
model.add(Embedding(input_dim=200001, output_dim=32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer
model.add(Dense(6, activation='sigmoid'))

```

<p>Embedding Layer</p>

<img src = "https://github.com/Vinit-joshi2/Comments-Toxicity/blob/main/Emedding%20Model.png">

---

## 🧪 Make Prediction
<p>Let's make a prediction on some random comment and check whether our model is working well or not</p>

<p>Input text</p>

```
input_text = vectorize("You freaking suck! I am goint to kill you")
```
<p> It's seem like comment is very toxic may threat someone</p>

<p>Now let's see whether our model detect this comment as a toxic or not?</p>

```
model.predict(np.expand_dims(input_text , 0))
```
<img src = "https://github.com/Vinit-joshi2/Comments-Toxicity/blob/main/img3.1.png">

<img src = "https://github.com/Vinit-joshi2/Comments-Toxicity/blob/main/img3.2.png"> 

<p>98% comment is toxic</p>



---
## 🧪 Evaluation Metrics

- Precision 
- Recall 
- CategoricalAccuracy

```
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
```



---

✅ Final Model Evaluation

| Metric        | Score |
| ------------- | ----- |
| **Precision** | 0.804 |
| **Recall**    | 0.722 |
| **Accuracy**  | 0.498 |



---

## 🔍 Example Predictions

| Comment                                              | Prediction |
|------------------------------------------------------|------------|
| "Thank you for your helpful feedback!"               | Non-Toxic  |
| "You're so dumb, this makes no sense."               | Toxic      |
| "I appreciate the detailed explanation."             | Non-Toxic  |
| "Nobody wants to hear your opinion, shut up."        | Toxic      |

---


