# Twitter Sentiment Analysis App

This project is a complete Natural Language Processing (NLP) pipeline that processes tweets and classifies their sentiment as **Positive**, **Negative**, or **Neutral** using various machine learning models. It ends with a user-friendly **Streamlit app** for real-time tweet classification.

---

## Features

- Data cleaning and preprocessing
- Tokenization
- Stopword removal
- Stemming and Lemmatization
- Named Entity Recognition (NER) *(optional enhancement)*
- Text Vectorization using `CountVectorizer`
- Model training using:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - Support Vector Machines
- Model evaluation (precision, recall, F1-score)
- Streamlit deployment

---

## Project Pipeline

### 1. Data Preprocessing

- Remove null values
- Convert text to lowercase
- Remove URLs, mentions, hashtags, digits, and punctuation

### 2. Tokenization

- Split text into words using `nltk.word_tokenize`

### 3. Stopword Removal

- Use NLTK's list of English stopwords

### 4. Stemming & Lemmatization

- Use `PorterStemmer` and `WordNetLemmatizer`

### 5. Vectorization

- Convert processed tokens into numerical features using `CountVectorizer`

### 6. Model Training & Evaluation

Models trained using `GridSearchCV` and/or `RandomizedSearchCV`:

- **Logistic Regression**
- **Random Forest**
- **Multinomial Naive Bayes**
- **Support Vector Machine (Linear SVC)**

### 7. Best Model Deployment

- The best-performing model was saved using `joblib` and deployed via **Streamlit**.

---

## Performance (Logistic Regression)

| Metric      | -1 (Negative) | 0 (Neutral) | 1 (Positive) |
|-------------|---------------|-------------|---------------|
| Precision   | 0.86          | 0.88        | 0.92          |
| Recall      | 0.80          | 0.95        | 0.89          |
| F1-Score    | 0.83          | 0.91        | 0.90          |

Overall Accuracy: **0.89**

---

## Streamlit App

The app takes in a tweet and classifies its sentiment in real-time.

```bash
streamlit run app.py
```

App UI:
- **Input**: Tweet text
- **Output**: Predicted sentiment label (Positive / Negative / Neutral)

---

## Libraries Used

- `pandas`, `numpy`, `re`, `string`
- `nltk`
- `scikit-learn`
- `joblib`
- `streamlit`

---

## File Structure

```
.
├── Twitter_Data.csv
├── skemers.py
├── Logistic_model.pkl
├── index.ipynb
├── Logistic_model.pkl
└── README.md
```

---

## Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/Godfrey-249/Phase-4-Group-2--NLP-Project.git
cd Phase-4-Group-2--NLP-Project
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Don't forget to download necessary NLTK packages:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## Notes

- Named Entity Recognition (NER) was considered for future enhancements.
- Model performance can be improved using `TF-IDF`, word embeddings (Word2Vec, BERT), or deep learning models (LSTM, BERT).

---

## Contact

Developed by: **Phase-4-Group-2**  
GitHub: [github.com/Godfrey-249](https://github.com/Godfrey-249/Phase)
