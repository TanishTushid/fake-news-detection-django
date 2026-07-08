"""
ML Training Pipeline for Fake News Detection.
- Loads data
- Preprocesses text
- Vectorizes via TF-IDF
- Trains multiple models
- Evaluates and saves the best model
"""
import os
import re
import string
import logging
import joblib
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ML_DIR = os.path.join(BASE_DIR, 'detector', 'ml')
os.makedirs(ML_DIR, exist_ok=True)

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' ', text)
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def main():
    logger.info("Starting ML Training Pipeline...")

    # 1. Load Data
    train_path = os.path.join(DATA_DIR, 'train.csv')
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}")
        return

    logger.info("Loading training data...")
    df = pd.read_csv(train_path)
    
    # Drop missing values
    df.dropna(subset=['Statement', 'Label'], inplace=True)
    logger.info(f"Dataset shape after dropping NA: {df.shape}")

    # 2. Preprocess Text
    logger.info("Preprocessing text data (this may take a minute)...")
    # For speed in this demonstration, we sample if dataset is huge, but here it's 10k so we do all
    df['clean_text'] = df['Statement'].apply(preprocess_text)
    
    # 3. Train-Test Split
    logger.info("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        df['clean_text'], df['Label'], test_size=0.2, random_state=42, stratify=df['Label']
    )

    # 4. Feature Engineering (TF-IDF)
    logger.info("Vectorizing text via TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 5. Train Multiple Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42),
        'Passive Aggressive': PassiveAggressiveClassifier(max_iter=50, random_state=42)
    }

    best_model = None
    best_acc = 0
    best_name = ""

    logger.info("Training and evaluating models...")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)
        acc = accuracy_score(y_val, preds)
        logger.info(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    logger.info(f"Best Model: {best_name} (Acc: {best_acc:.4f})")
    
    # Show detailed classification report for best model
    val_preds = best_model.predict(X_val_vec)
    logger.info("\nClassification Report:\n" + classification_report(y_val, val_preds))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_val, val_preds)))

    # 6. Save Model and Vectorizer
    model_path = os.path.join(ML_DIR, 'model.pkl')
    vec_path = os.path.join(ML_DIR, 'vectorizer.pkl')
    
    logger.info("Saving best model and vectorizer...")
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved vectorizer to {vec_path}")
    
    logger.info("Pipeline Complete!")

if __name__ == '__main__':
    main()
