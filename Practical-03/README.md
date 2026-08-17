# 🧪 Practical 03: Word Embeddings, Vector Arithmetic & Similarity Visualizations

A comprehensive laboratory implementation covering distributed word representations using **Word2Vec (Gensim)**, exploring model architectures (**CBOW vs. Skip-gram**), vector arithmetic for semantic analogies, dimensionality reduction via **PCA**, and interactive matrix visualization with **Plotly**.

---

## 📌 Objectives & Key Concepts

1. **Custom Word2Vec Training:** Implemented Continuous Bag of Words (`sg=0`) and Skip-gram (`sg=1`) architectures on domain corpora.
2. **Pre-trained Embeddings:** Evaluated semantic relationships using Google News 300-dimensional vectors (`word2vec-google-news-300`).
3. **Vector Arithmetic & Analogies:** Demonstrated algebraic relationship properties in embedding spaces:
   $$\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$$
4. **2D Dimensionality Reduction:** Applied Principal Component Analysis (**PCA**) to project 300D embedding vectors into a 2D coordinate plane to observe semantic clustering across categories (royalty, gender, fruits, vehicles, emotions).
5. **Interactive Similarity Heatmap:** Generated pairwise cosine similarity matrices visualized through **Plotly**.

---

## 🛠️ Prerequisites & Installation

```bash
pip install gensim scikit-learn matplotlib plotly pandas numpy
```

---

## 🔬 Implementation Summary

### 1. Vector Arithmetic & Analogies
```python
# Analogy: King - Man + Woman = Queen
similar_words = model.most_similar(
    positive=['king', 'woman'], 
    negative=['man'], 
    topn=5
)

for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

**Results:**
```text
queen: 0.7118
monarch: 0.6190
princess: 0.5902
crown_prince: 0.5499
prince: 0.5377
```

---

### 2. Pairwise Cosine Similarity Heatmap (Plotly)
```python
import plotly.express as px
import pandas as pd

words = ['King', 'Queen', 'Man', 'Woman']
similarity_matrix = pd.DataFrame(index=words, columns=words, dtype=float)

for i in range(len(words)):
    for j in range(len(words)):
        similarity_matrix.iloc[i, j] = model.similarity(words[i], words[j])

fig = px.imshow(
    similarity_matrix,
    text_auto=".4f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Word Similarity Matrix"
)
fig.show()
```

---

## 📊 Observations

* **Architecture Trade-offs:** CBOW optimizes faster over frequent vocabulary tokens, whereas Skip-Gram models capture granular semantic distinctions and rare terms effectively.
* **Vector Algebra:** The linear compositionality of dense word vectors preserves abstract conceptual relationships such as gender and status across semantic domains.

---

## 👤 Author

* **Name:** Shreya Nimje
* **Program:** MCA (Artificial Intelligence & Machine Learning)
* **Practical:** 03 - Word Embeddings & Vector Space Models
