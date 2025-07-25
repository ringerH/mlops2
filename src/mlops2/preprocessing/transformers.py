import re
from typing import List
from emoji import replace_emoji
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- 1.  Acronym expansion ----------
_ACRONYMS = {
    "FOMO":"Fear Of Missing Out","FUD":"Fear Uncertainty Doubt","DYOR":"Do Your Own Research",
    "BTFD":"Buy The Fear Dip","HODL":"Hold On For Dear Life","ATH":"All Time High",
    "ATL":"All Time Low","IPO":"Initial Public Offering","ROI":"Return On Investment",
    "EPS":"Earnings Per Share","P/E":"Price To Earnings Ratio","YTD":"Year To Date",
    "YOY":"Year Over Year","QoQ":"Quarter Over Quarter","SL":"Stop Loss","TP":"Take Profit",
    "PT":"Price Target","MCAP":"Market Capitalization","VOL":"Trading Volume",
    "ETF":"Exchange Traded Fund","CFD":"Contract For Difference","MOON":"To The Moon",
    "BEAR":"Bearish Sentiment","BULL":"Bullish Sentiment",
}
_ACRO_PAT = re.compile(r"\b(" + "|".join(map(re.escape, _ACRONYMS)) + r")\b", re.I)

class AcronymExpander(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return [_ACRO_PAT.sub(lambda m: _ACRONYMS[m.group(1).upper()], txt) for txt in X]

# ---------- 2.  HTML / emoji stripper ----------
def strip_html_and_emoji(texts: List[str]) -> List[str]:
    cleaned = []
    for t in texts:
        no_html  = re.sub(r"&lt;.*?&gt;", " ", t)
        no_emoji = replace_emoji(no_html, replace="")
        cleaned.append(no_emoji)
    return cleaned

# ---------- 3.  spaCy cleaner ----------
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class SpacyCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        out = []
        for doc in _nlp.pipe(X, batch_size=32):
            toks = [
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and not tok.like_url and not tok.like_email
            ]
            out.append(" ".join(toks))
        return out
