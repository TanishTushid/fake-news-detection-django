"""
Utility functions for the detector app.

Responsibilities:
- Text preprocessing (mirrors the training pipeline exactly)
- Loading the ML model and vectorizer (singleton pattern)
- Running inference and returning label + confidence
"""
import re
import string
import logging
import joblib
import numpy as np

from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap — download required corpora on first run
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    def _ensure_nltk_data():
        packages = ['stopwords', 'wordnet', 'omw-1.4']
        for pkg in packages:
            try:
                nltk.data.find(f'corpora/{pkg}')
            except LookupError:
                nltk.download(pkg, quiet=True)

    _ensure_nltk_data()
    STOP_WORDS = set(stopwords.words('english'))
    _lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.warning("NLTK not fully available: %s. Falling back to basic preprocessing.", exc)
    STOP_WORDS = set()
    NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Apply the same preprocessing pipeline used during model training:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove punctuation
      5. Remove digits / numbers
      6. Remove extra whitespace
      7. Remove stopwords
      8. Lemmatize (if NLTK available)
    """
    if not text or not isinstance(text, str):
        return ''

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 3. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Remove digits
    text = re.sub(r'\d+', ' ', text)

    # 6. Tokenise (simple whitespace split)
    tokens = text.split()

    # 7. Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # 8. Lemmatize
    if NLTK_AVAILABLE:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Model / Vectorizer Loading  (singleton — loaded once per process)
# ---------------------------------------------------------------------------

_model = None
_vectorizer = None


def _load_artifacts():
    """Load model and vectorizer from disk (lazy singleton)."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        model_path = settings.ML_MODEL_PATH
        vec_path = settings.ML_VECTORIZER_PATH
        try:
            _model = joblib.load(model_path)
            _vectorizer = joblib.load(vec_path)
            logger.info("ML artifacts loaded from %s and %s", model_path, vec_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Model artifacts not found. Please run `python train_model.py` first."
            ) from exc
    return _model, _vectorizer


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(raw_text: str) -> dict:
    """
    Preprocess raw_text, run inference, and return a result dict.

    Returns:
        {
            'label': 'REAL' | 'FAKE',
            'confidence': float (0.0 – 1.0),
            'preprocessed': str,
        }
    """
    model, vectorizer = _load_artifacts()

    preprocessed = preprocess_text(raw_text)
    if not preprocessed.strip():
        return {'label': 'FAKE', 'confidence': 0.5, 'preprocessed': preprocessed}

    X = vectorizer.transform([preprocessed])

    # Probability scores (if the model supports predict_proba)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        predicted_class = model.classes_[np.argmax(proba)]
        confidence = float(np.max(proba))
    else:
        # Fallback: use decision_function for models like LinearSVC
        decision = model.decision_function(X)[0]
        # Convert to a sigmoid-style confidence
        confidence = float(1 / (1 + np.exp(-abs(decision))))
        predicted_class = model.predict(X)[0]

    # The model was trained with True=Real, False=Fake
    # predicted_class may be True/False (bool) or 'True'/'False' (str)
    if isinstance(predicted_class, bool):
        label = 'REAL' if predicted_class else 'FAKE'
    elif str(predicted_class).lower() in ('true', '1'):
        label = 'REAL'
    else:
        label = 'FAKE'

    return {
        'label': label,
        'confidence': confidence,
        'preprocessed': preprocessed,
    }
