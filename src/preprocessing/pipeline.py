from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from .transformers import AcronymExpander, SpacyCleaner, strip_html_and_emoji


def create_preprocessing_pipeline() -> Pipeline:
    """Returns an sklearn pipeline that converts raw text → TF-IDF matrix."""
    return Pipeline([
        ("strip",  FunctionTransformer(strip_html_and_emoji, validate=False)),
        ("acro",   AcronymExpander()),
        ("clean",  SpacyCleaner()),
        ("tfidf",  TfidfVectorizer(tokenizer=str.split, preprocessor=lambda x: x,
                                   token_pattern=None, min_df=1, max_df=1.0)),
    ])
