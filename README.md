# 📰 Fake News Detector

> **Explainable machine learning for detecting potentially fake or real news using NLP and classical machine learning.**

The **Fake News Detector** is an end-to-end machine learning project that demonstrates how natural language processing and supervised learning can be used to classify news articles as **FAKE** or **REAL**.

The project combines an NLP preprocessing pipeline, TF-IDF feature extraction, multiple machine-learning classifiers, model comparison, prediction explanations, batch scoring, and a Streamlit dashboard.

🌐 **Repository:**
https://github.com/swapnil1222589/Fake-News-Detector-Project-in-AI
 
---

## 🚀 Project Overview

Misinformation can spread rapidly through online platforms, making automated text classification useful as a research and educational tool.

This project explores a complete ML workflow:

```text
News Article
     ↓
Text Cleaning
     ↓
NLTK NLP Pipeline
     ↓
TF-IDF Feature Extraction
     ↓
Multiple ML Models
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Prediction + Confidence
     ↓
Token-Level Explanation 
```

The repository is designed as a **portfolio-grade ML project**, emphasizing reproducibility, modularity, evaluation, and explainability.

> ⚠️ This system provides a machine-learning classification result, not a definitive determination of whether a real-world news claim is true. Model predictions can be affected by the dataset, training process, and characteristics of the input article.

---

## ✨ Features

### 📰 Fake / Real Classification

Classify an input news article into:

* `FAKE`
* `REAL`

with an associated confidence score.

### 🧹 NLP Preprocessing

The project includes a reusable text-cleaning pipeline covering:

* Lowercasing
* URL removal
* HTML removal
* Punctuation removal
* Tokenization
* Stopword removal
* WordNet lemmatization

### 🔢 TF-IDF Feature Engineering

Cleaned text is transformed into numerical vectors using TF-IDF.

The showcased training configuration uses:

```text
50,000 features
1–2 word n-grams
Sublinear TF scaling
```

### 🧪 Multiple Model Comparison

Five classifiers are compared:

* Logistic Regression
* Passive Aggressive Classifier
* Multinomial Naive Bayes
* Random Forest
* Linear SVM

The training workflow automatically identifies the best-performing model based on F1 score.

### 🔍 Explainable Predictions

The inference layer can surface important contributing tokens so users can inspect which words had strong influence on the model's decision.

### 📦 Batch Prediction

Score multiple articles from a CSV file containing a `text` column.

### 📊 Performance Dashboard

The Streamlit interface provides model performance information including:

* Accuracy
* Precision
* Recall
* F1 score
* Model comparison

### 📈 Evaluation Artifacts

The project supports generating:

* Confusion matrix
* Classification report
* ROC curve

### 🧾 Prediction History

Predictions made during a session can be reviewed and exported as CSV.

### 🌑 Dark-Mode Dashboard

The project includes a modern dark-themed interface suitable for demos and portfolio presentations.

---

## 📊 Model Results

The repository's documented experiment compares five models:

| Model                   |   Accuracy |  Precision |     Recall |         F1 |
| ----------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression     |     0.9878 |     0.9871 |     0.9887 |     0.9879 |
| Passive Aggressive      | **0.9954** | **0.9951** | **0.9959** | **0.9955** |
| Multinomial Naive Bayes |     0.9382 |     0.9330 |     0.9473 |     0.9401 |
| Random Forest           |     0.9911 |     0.9903 |     0.9921 |     0.9912 |
| Linear SVM              |     0.9948 |     0.9944 |     0.9954 |     0.9949 |

The repository's displayed benchmark identifies **Passive Aggressive** as the highest-F1 model in that experiment.

> Metrics depend on the specific dataset split, preprocessing, feature configuration, and training environment. They should not be interpreted as universal real-world accuracy.

---

## 🧠 Machine Learning Pipeline

```text
Raw News Article
       ↓
URL / HTML Removal
       ↓
Lowercase
       ↓
Punctuation Cleaning
       ↓
Tokenization
       ↓
Stopword Removal
       ↓
Lemmatization
       ↓
TF-IDF Vectorization
       ↓
┌─────────────────────────────┐
│ Logistic Regression         │
│ Passive Aggressive          │
│ Multinomial Naive Bayes     │
│ Random Forest               │
│ Linear SVM                  │
└──────────────┬──────────────┘
               ↓
       Model Evaluation
               ↓
        Best Model by F1
               ↓
          Saved Model
               ↓
        Prediction Layer
               ↓
      Streamlit Application
```

---

## 🛠️ Technology Stack

### Programming

* Python 3.12+

### Machine Learning

* Scikit-learn
* NumPy
* Pandas
* Joblib

### Natural Language Processing

* NLTK
* WordNet
* Tokenization
* Stopword filtering
* Lemmatization

### Visualization & Evaluation

* Matplotlib
* Seaborn

### Application

* Streamlit

### Development

* Git
* GitHub
* Pytest
* Ruff

---

## 📂 Project Structure

```text
Fake-News-Detector-Project-in-AI/
├── app.py
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metrics.json
│
├── notebooks/
│   └── model_training.ipynb
│
├── docs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── Makefile
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📁 Core Components

### `app.py`

Streamlit entry point responsible for:

* Single article prediction
* Batch prediction
* Prediction history
* Confidence visualization
* Model dashboard
* CSV exports

### `src/preprocess.py`

Contains the reusable NLP cleaning pipeline.

### `src/train.py`

Responsible for:

* Dataset loading
* Train/test splitting
* TF-IDF feature extraction
* Training multiple classifiers
* Model evaluation
* Best-model selection
* Artifact persistence

### `src/predict.py`

Provides the prediction layer used by the application, including:

* Classification
* Confidence estimation
* Probability information
* Token explanations
* DataFrame/batch scoring

### `src/evaluate.py`

Generates machine-learning evaluation artifacts such as:

* Confusion matrix
* ROC curve
* Classification report

### `src/utils.py`

Centralizes:

* Project paths
* Logging
* Metrics loading
* Model artifact locations

### `tests/test_pipeline.py`

Provides automated tests for preprocessing and input validation.

---

## 📚 Dataset

The project documentation references the **ISOT Fake News Dataset**, containing separate fake and real news collections.

The showcased dataset summary is:

```text
Fake articles:  23,481
Real articles:  21,417
Total:          44,898
```

The project combines article title/body content into a text representation for classification.

### Dataset Files

```text
data/
├── Fake.csv
└── True.csv
```

Dataset source:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

> Dataset usage and redistribution should follow the terms of the original dataset source.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/swapnil1222589/Fake-News-Detector-Project-in-AI.git
cd Fake-News-Detector-Project-in-AI
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Models

The training workflow can be launched with:

```bash
python -m src.train --data-dir data --models-dir models
```

The process:

1. Loads the dataset
2. Cleans the text
3. Builds TF-IDF features
4. Trains five classifiers
5. Calculates evaluation metrics
6. Compares model performance
7. Selects the best model
8. Saves model artifacts

Expected artifacts include:

```text
models/
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── metrics.json
```

---

## ▶️ Run the Streamlit App

Start the application:

```bash
streamlit run app.py
```

The Streamlit server will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🧪 Batch Prediction

The batch prediction interface accepts a CSV containing a column named:

```text
text
```

Example:

```csv
text
"News article content goes here..."
"Another article for classification..."
```

The application can then produce predictions and confidence values and export the scored dataset as CSV.

---

## 🔍 Explainability

One of the project's key goals is making predictions easier to inspect.

For linear-model predictions, the application can surface influential tokens associated with the model's decision.

Conceptually:

```text
Article
   ↓
TF-IDF Features
   ↓
Model Weights
   ↓
Important Tokens
   ↓
Prediction Explanation
```

This helps users understand that the prediction is based on learned textual patterns rather than being presented as an unexplained label.

---

## 📈 Evaluation

The project includes evaluation tooling for:

### Confusion Matrix

Shows:

```text
Actual vs Predicted
REAL / FAKE
```

### Classification Report

Reports:

* Precision
* Recall
* F1
* Support

### ROC Curve

Provides an additional view of classifier discrimination performance.

---

## ✅ Testing

Run the test suite with:

```bash
pytest -q
```

The repository includes tests covering preprocessing behavior and input validation.

---

## 🧹 Code Quality

The project is structured for maintainability with:

* Type hints
* Logging
* Modular source files
* Error handling
* Automated tests
* Centralized configuration paths

When Ruff is installed, checks can be run with:

```bash
ruff check src app.py
```

---

## ⚡ Makefile Commands

The repository includes convenience commands for common workflows:

```bash
make setup
make lint
make test
make train
make run
make clean
```

---

## 🎯 Project Objectives

This project was built to demonstrate practical understanding of:

* NLP
* Text classification
* Feature engineering
* Supervised machine learning
* Model comparison
* Explainable predictions
* Streamlit application development
* ML project organization
* Testing and code quality

---

## 🔮 Future Improvements

Potential next steps include:

### 🤖 Advanced NLP

* DistilBERT baseline
* Transformer-based classifiers
* Context-aware embeddings
* Fine-tuned language models

### 🔍 Better Explainability

* SHAP integration
* LIME explanations
* Interactive token highlighting

### 🌐 Production API

* FastAPI inference service
* REST API
* Docker container
* Cloud deployment

### 📊 MLOps

* CI/CD with GitHub Actions
* Experiment tracking
* Model versioning
* Data drift monitoring
* Automated retraining

### 🛡️ Reliability

* Better confidence calibration
* Out-of-distribution detection
* Adversarial-text testing
* More diverse datasets
* Cross-dataset evaluation

---

## ⚠️ Limitations

A machine-learning fake-news detector has important limitations.

The model learns patterns from its training data. Consequently:

* It may perform differently on unseen sources.
* Changes in writing style can affect predictions.
* Dataset bias can affect model behavior.
* High confidence does not guarantee factual correctness.
* A model trained on historical data may not understand current events.

For real-world fact checking, model output should be treated as **one signal among several**, alongside source verification and independent evidence.

---

## 🌟 Why This Project Matters

The project demonstrates an end-to-end ML workflow rather than only a trained classifier.

It connects:

```text
Data
 ↓
NLP
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Selection
 ↓
Evaluation
 ↓
Explainability
 ↓
Application
 ↓
Testing
```

That makes it useful as a practical example of taking an ML experiment toward a usable application.

---

## 🤝 Contributing

Contributions are welcome.

```bash
git checkout -b feature/your-feature
```

Make your changes, test them, and open a pull request.

Please include a clear description of:

* What changed
* Why it changed
* How it was tested

---

## 📄 License

This project includes an MIT license.

See [`LICENSE`](LICENSE) for details.

---

## 👨‍💻 Author

**Swapnil Ghuge**

GitHub:
https://github.com/swapnil1222589

Repository:
https://github.com/swapnil1222589/Fake-News-Detector-Project-in-AI

---

## ⭐ Support

If you find this project useful for learning NLP, machine learning, or MLOps, consider giving the repository a ⭐.

> **Detect patterns. Understand the model. Verify the facts.**
