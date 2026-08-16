# 🔬 Practical 01: End-to-End NLP Text Analyzer

An interactive web application built with **Streamlit**, **spaCy**, and **NLTK** that performs complete Natural Language Processing (NLP) text analysis, including lexical normalization, syntactic parsing, and information extraction.

---

## 📌 Features & Pipeline

1. **Sentence Segmentation & Word Tokenization:** Splits paragraphs into clean sentence boundaries and individual linguistic tokens using NLTK's `punkt`.
2. **Stop Word Removal:** Filters common non-informative English stop words.
3. **Morphological Comparison:** Side-by-side comparison between **Porter Stemmer** (heuristic-based) and **WordNet Lemmatizer** (lexical/vocabulary-based).
4. **Part-of-Speech (POS) Tagging:** Assigns Penn Treebank tags with full human-readable descriptions.
5. **Named Entity Recognition (NER):** Extracts structured real-world entities (PERSON, ORG, GPE, DATE, etc.) using spaCy.
6. **Dependency Parsing:** Extracts head-child grammatical relationships and dependency tags.

---

## 🛠️ Installation & Dependencies

Ensure you have Python installed, then install the required dependencies:

```bash
pip install streamlit nltk spacy pandas
```

Download the required spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 How to Run

1. Navigate to the `Practical-02` directory:
   ```bash
   cd Practical-02
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run Pr2.py
   ```

3. Open your browser at `http://localhost:8501`.

---

## 🧪 Sample Test Input

```text
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in California. The company is actively designing revolutionary smartphones, tablets, and computers. Tim Cook visited Mumbai yesterday to inaugurate two new retail stores.
```

---

## 👤 Author

* **Name:** Shreya Nimje
* **Program:** MCA (Artificial Intelligence & Machine Learning)
* **Practical:** 02 - Natural Language Processing
