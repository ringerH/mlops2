from pathlib import Path
import pandas as pd
import tempfile

from ml-ops.data.loader import DataLoader
from ml-ops.config.settings import DATA_CONFIG

def test_csv_loading(tmp_path):
    # create temp CSV
    tmp_csv = tmp_path / "data.csv"
    df = pd.DataFrame({"Sentence":["hi"],"Sentiment":["positive"]})
    df.to_csv(tmp_csv, index=False)

    loader = DataLoader(data_dir=tmp_path)
    out_df = loader.load_data()
    assert out_df.equals(df)
    # schema check
    assert set(out_df.columns) == {DATA_CONFIG["text_column"], DATA_CONFIG["target_column"]}
