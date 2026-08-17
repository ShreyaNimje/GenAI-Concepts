# 🔬 Practical 04: Next-Word Prediction using Recurrent Architectures (SimpleRNN, LSTM, GRU)

A comparative implementation of sequence modeling architectures using **TensorFlow/Keras** to perform character/word-level text tokenization, N-gram sequence generation, and real-time next-word prediction[cite: 1].

---

## 📌 Overview & Methodology

1. **Text Preprocessing & Tokenization:** Processes raw corpus data (`data.txt`) using Keras `Tokenizer` and transforms text into sequential token integers[cite: 1].
2. **N-gram Sequence Generation:** Generates contiguous sub-sequences ($N$-grams) to create supervised training pairs ($X \rightarrow y$)[cite: 1].
3. **Pre-Padding:** Normalizes input sequence lengths using `pad_sequences(..., padding='pre')`[cite: 1].
4. **Architecture Comparison:** Evaluates training dynamics, convergence, and inference behavior across three core recurrent topologies[cite: 1]:
   * **SimpleRNN (Baseline)**[cite: 1]
   * **LSTM (Long Short-Term Memory)**[cite: 1]
   * **GRU (Gated Recurrent Unit)**[cite: 1]
5. **Interactive Prediction Loop:** Provides an interactive command-line interface to forecast subsequent tokens given user prompt context[cite: 1].

---

## 🏗️ Model Architectures

* **SimpleRNN:** Embedding layer ($10\text{D}$) $\rightarrow$ `SimpleRNN(50)` $\rightarrow$ `Dense(Softmax)`[cite: 1].
* **LSTM:** Embedding layer ($10\text{D}$) $\rightarrow$ `LSTM(150)` $\rightarrow$ `Dense(Softmax)`[cite: 1].
* **GRU:** Embedding layer ($10\text{D}$) $\rightarrow$ `GRU(150)` $\rightarrow$ `Dense(Softmax)`[cite: 1].

---

## 📊 Training Performance & Comparison

All architectures were trained using the `adam` optimizer with `sparse_categorical_crossentropy` loss across **100 epochs**[cite: 1]:

| Model Architecture | Hidden Units | Final Loss (Epoch 100) | Execution Speed |
| :--- | :--- | :--- | :--- |
| **SimpleRNN** | 50 | **~0.5902**[cite: 1] | ~11–12 ms/step[cite: 1] |
| **LSTM** | 150 | **~0.5097**[cite: 1] | ~50–60 ms/step[cite: 1] |
| **GRU** | 150 | **~0.2142**[cite: 1] | ~55–65 ms/step[cite: 1] |

---

## 🛠️ Installation & Setup

```bash
pip install tensorflow numpy
```

Ensure `data.txt` is present in the working directory before running[cite: 1]:

```text
Practical-04/
├── data.txt                # Input corpus
├── RNN.ipynb               # Colab / Jupyter Notebook
└── README.md               # Documentation
```

---

## 🧪 Interactive Sample Output

```text
Enter text (or 'quit' to exit): big
Predicted next word: played

Enter text (or 'quit' to exit): Ria
Predicted next word: and

Enter text (or 'quit' to exit): Thank you so much
Predicted next word: ”
```
[cite: 1]

---

## 👤 Author

* **Name:** Shreya Nimje
* **Program:** MCA (Artificial Intelligence & Machine Learning)
* **Practical:** 04 - Recurrent Neural Networks (SimpleRNN, LSTM, GRU)
