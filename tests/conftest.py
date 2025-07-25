import pandas as pd
import pytest

@pytest.fixture(scope="session")
def sample_df():
    data = [
        {"Sentence":"I love this stock! HODL 🚀",             "Sentiment":"positive"},
        {"Sentence":"Terrible results, full of FUD",          "Sentiment":"negative"},
        {"Sentence":"Fear Of Missing Out on ATH breakout",    "Sentiment":"positive"},
        {"Sentence":"Worst quarter ever… SELL!","Sentiment":"negative"},
    ]
    return pd.DataFrame(data)
